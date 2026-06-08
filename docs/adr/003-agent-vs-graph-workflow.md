# ADR-003: Agent 循环 vs LangGraph 工作流 — 双轨架构

**状态**: 已采纳  
**日期**: 2025-06  
**决策者**: @yiyao1226

---

## Context（背景）

LLM 应用有两种主流交互模式：

**Agent 循环（Autonomous Agent）**：
LLM 自主决定调用哪些工具、调用顺序、何时停止。
参考 Claude Code 的 agent loop：think → tool_call → observe → think → ...

**工作流（Workflow/Chain）**：
预先定义节点和边，LLM 在每个节点内执行特定任务。
参考 LangGraph：StateGraph → add_node → add_edge → compile。

两者各有优劣：
- Agent 灵活但不可控：LLM 可能调用过多工具、走偏方向、无限循环
- Workflow 可控但不灵活：固定流程无法处理意外情况

我们的系统有两种典型场景：
- **日常问答**："CVD 温度多少？" → 需要灵活决定搜不搜、怎么搜
- **文献综述**："写一篇金刚石薄膜的综述" → 确定性流程：搜索→分析→综合→审查

## Decision（决策）

采用**双轨架构**，按场景选择：

```
场景判断: 用户意图
  │
  ├─ 闲聊 / 概念解释 / 简单科研问题
  │    → Agent 循环 (qa.py:agent_loop)
  │       LLM 自主决定: 不调工具 / 调 query_knowledge_base / 调 search_papers_online
  │       最多 3 轮工具调用
  │
  └─ 文献综述 / 深度研究 / 进展评估
       → LangGraph 工作流 (graph.py:build_research_graph)
          4 节点: understand → research(可迭代) → synthesize → user_review
          每个节点有 checkpoint，可中断/恢复/人工干预
```

### Agent 循环设计（qa.py）

```
agent_loop(messages, tools, llm, max_rounds=3):
  for round in range(max_rounds):
    response = llm.invoke(messages)
    if has_tool_calls:
      for tc in response.tool_calls:
        result = exec_tool(tc)
        messages.append(ToolMessage(result))
      continue  # 下一轮
    else:
      _stream_answer(response)  # 最终回答
      break
```

关键设计：
- **invoke 检测工具调用**（不用 stream）→ 可靠判断 tool_calls 存在性
- **最终回答本地逐字输出**（不再重复调 LLM）→ 节省 token
- **Hook 系统**：pre_tool / post_tool / before_llm → 横切关注点（日志、降级、后台派发）

### LangGraph 工作流设计（graph.py）

```
                          ┌─────────────┐
                          │ understand   │  盘点 KB + 制定搜索策略
                          │ (Agent + KB  │
                          │  tools only) │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                    ┌─────│  research    │  搜索论文 + 自评估
                    │     │ (Agent + all │  ← 可迭代最多 3 轮
                    │     │  tools)      │
                    │     └──────┬──────┘
                    │            │
                    │     agent_satisfied?
                    │        │         │
                    │       NO        YES
                    │        │         │
                    └────────┘    ┌────▼───────┐
                                  │ synthesize  │  综合撰写/回答
                                  │ (纯 LLM，    │
                                  │  无工具)     │
                                  └────┬───────┘
                                       │
                                  ┌────▼───────┐
                                  │ user_review │  人机协同
                                  │ (interrupt) │  ← 可中断
                                  └────┬───────┘
                                       │
                                  确认/修改?
                                   │      │
                                  END  synthesize
```

关键设计：
- **research 可迭代**：Agent 不满意就换搜索策略重搜
- **interrupt 人机协同**：最终结果让人确认，修改意见回流到 synthesize 重生成
- **SqliteSaver checkpoint**：每个节点完成后持久化状态，崩溃后可恢复

## Consequences（后果）

### 正面
- Agent 循环适合开放域问答 — 灵活、Token 高效
- LangGraph 适合确定性流程 — 可控、可恢复、可审计
- 两者共享相同的工具集（`make_all_agent_tools`）— 无代码重复
- Checkpoint 机制让长流程（5-10 分钟综述）不丢进度

### 负面
- 双轨维护成本：两套 prompt 模板、两种状态管理
- Agent 循环的 `max_rounds=3` 是硬限制 — 复杂问题可能不够
- LangGraph 的 `interrupt` 在 CLI 里是 `input()` — 阻塞主线程

### 风险
- research 节点的 "满意" 判断是关键词匹配（"满意" in text）— 可能误判
- user_review 的修改意见直接拼接到 topic — 可能产生奇怪的 prompt
- SqliteSaver 在并发场景有锁竞争（单用户 CLI 无此问题）

## Alternatives Considered

### A. 全用 Agent 循环
- **否决原因**：综述需要 6-8 步确定性操作，Agent 自由度过高容易跑偏。
  而且 Agent 中途崩溃没有 checkpoint 可恢复。

### B. 全用 LangGraph
- **否决原因**："今天天气怎么样" 这种闲聊走 4 节点工作流太重了。
  简单问答应该是一个 LLM 调用完成，不需要状态机。

### C. 用 CrewAI/AutoGen 多 Agent
- **否决原因**：多 Agent 在这个规模下是过度工程。
  一个 Agent 循环 + 一个 Workflow 已经覆盖全场景。
  而且多 Agent 的 token 爆炸是实打实的成本问题。

---

## 面试要点

- **核心概念**：不是所有问题都需要 Agent — 区分"开放域决策"和"确定性流程"
- **判断标准**：用户意图评估 → 闲聊=Agent, 综述=Workflow
- **工程细节**：checkpoint 的价值 — 崩溃恢复、人工干预点
- **trade-off**：灵活性 vs 可控性，根据场景而不是一刀切
