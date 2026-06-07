# 科研助手 — 深度分析笔记

> 面试用。前三轮：Agent 循环、LangGraph 工作流、记忆系统。

---

## 一、Agent 循环架构

### 1.1 整体调用链

```
main.py → QAService.ask()
  ├─ make_all_agent_tools()     → 5个 @tool（闭包捕获 storage + username）
  ├─ _build_context()           → 用户画像+工作记忆+进展+未解决问题 (~2300t)
  ├─ build_system_prompt()     → 工具描述从 @tool docstring 自动提取
  ├─ agent_loop()               → 可见循环体 + Hook 横切
  └─ add_turn()                 → 启动30s防抖定时器
```

### 1.2 Hook 系统

文件：[agent/qa.py:27-47](research_assistant/agent/qa.py#L27-L47)

6 个事件：`agent_start`、`before_llm`、`after_llm`、`pre_tool`、`post_tool`、`agent_end`。

`pre_tool` 有拦截语义——回调返回非 `None` 字符串时，工具真实逻辑被跳过，Hook 返回值伪装成 ToolMessage 发给 LLM。

内置 4 个 Hook（[qa.py:53-85](research_assistant/agent/qa.py#L53-L85)）：
- `_progress_hook` → agent_start → 打印"🤖 分析:..."
- `_tool_log_hook` → pre_tool → 打印工具调用（返回 None，不拦截）
- `_tool_result_hook` → post_tool → 打印结果摘要
- `_agent_done_hook` → agent_end → 打印统计

我们新增了 3 个 Hook：
- `_qdrant_fallback_hook` → pre_tool → Qdrant 离线时返回降级文本（拦截）
- `_bg_notification_hook` → before_llm → 注入已完成的后台任务结果
- `_bg_dispatch_hook` → pre_tool → 慢操作路由到后台线程（拦截）

### 1.3 agent_loop 两阶段策略

文件：[agent/qa.py:145-203](research_assistant/agent/qa.py#L145-L203)

```python
for round_num in range(max_rounds=3):     # 最多3轮工具调用
    response = llm.invoke(messages)        # invoke: 需要完整 tool_calls 才能执行
    if has_tools:
        for tc in response.tool_calls:
            blocked = trigger_hooks("pre_tool", name, args)  # Hook 可拦截
            result = str(blocked) if blocked else _exec_tool(tc, tools)
            messages.append(ToolMessage(result))
        continue  # 回到循环头，继续检测
    else:
        _stream_answer()  # stream: 逐 token 打字机效果
        break
```

工具阶段用 `invoke`（需要完整 tool_calls 解析 name/args/id），回答阶段用 `stream`（逐 token 输出用户体验好）。

### 1.4 System Prompt 动态组装

文件：[agent/qa.py:125-138](research_assistant/agent/qa.py#L125-L138)

```python
def build_system_prompt(tools, context):
    for t in tools:
        desc = t.description.split("\n")[0]  # ← 从 @tool docstring 自动提取
        tool_lines.append(f"{i}. **{t.name}** — {desc}")
    return AGENT_SYSTEM_TEMPLATE.format(tools=..., context=...)
```

工具描述和代码同源，增删工具不需要手动改 prompt。决策原则（何时调哪个工具）写死在模板里，降低 LLM 乱调工具的概率。

### 1.5 上下文压缩三段式

文件：[agent/compaction.py](research_assistant/agent/compaction.py)

| 级别 | 触发条件 | 策略 | API 调用 |
|------|---------|------|---------|
| L1 | 单条 tool_result > 30KB | 存盘留 2000 字预览 | 0 次 |
| L2 | 旧 tool_result 超过 4 个 | → `[已压缩]` 占位符 | 0 次 |
| L3 | 总 token 估算 > 50K | LLM 总结 → 摘要 + 最近 5 条 | 1 次 |

便宜的先执行。L1/L2 纯规则零 API 成本，L3 只在真正超阈值时触发。实际正常对话基本不会触发压缩——30KB/50K 阈值设得很保守。

### 1.6 Qdrant 离线降级

文件：[agent/qa.py:90-162](research_assistant/agent/qa.py#L90-L162)（我们实现的）

保护 4 个依赖 Qdrant 的工具：`query_knowledge_base`、`query_progress`、`recall_history`、`ingest_papers`。`search_papers_online` 走外网 API 不受影响。健康检查 30s TTL 缓存，避免每次工具调用都探测。

### 1.7 后台任务管理器

文件：[agent/background.py](research_assistant/agent/background.py)（我们实现的）

```
BackgroundTaskManager
├─ dispatch()           → ThreadPoolExecutor 派发，立即返回 task_id
├─ collect_notifications() → 收集完成的任务 → <task_notification> 格式
├─ shutdown(timeout=10s) → 优雅关闭 + interrupted 状态存 SQLite
└─ resume_interrupted() → 读取 SQLite 中上次中断的任务
```

慢操作识别：`search_papers_online`（~8s）、`ingest_papers`（~10-60s）走后台。快速工具同步执行不变。

---

## 二、LangGraph 工作流

### 2.1 核心概念

LangGraph = 有状态的有向图执行框架。三个核心概念：

- `StateGraph(状态类)` — 定义所有节点共享的数据结构
- `.add_node("名", 函数)` + `.add_edge("A", "B")` + `.add_conditional_edges(...)` — 拓扑
- `.compile(checkpointer=SqliteSaver)` — 每个节点完成后自动持久化状态到 SQLite

### 2.2 序列化与 RuntimeContext

**序列化** = 把内存里的 Python 对象变成能存盘的 JSON 字符串。

`ResearchState` 全是 JSON 能表示的类型（str/int/list/bool），SqliteSaver 能直接存。
`storage`（数据库连接）和函数指针无法序列化——它们是"活着的东西"（OS 文件句柄）。

**RuntimeContext** = 线程安全的旁路字典，通过 `thread_id` 隔离：

```python
# CLI 注入（graph.stream 之前）
set_runtime_context(thread_id, _storage=storage, _username="yiao1226")

# 节点内部获取
ctx = get_runtime_context(thread_id)
storage = ctx["_storage"]
```

节点崩溃后 RuntimeContext 丢失，但没关系——重新执行时 CLI 会再次注入。

### 2.3 SqliteSaver 的自动持久化

每个节点返回后，SqliteSaver 自动把整个 `ResearchState` 写入 SQLite 新一行。形成快照链：

```
checkpoint_0: understand 完成 → papers_found: []
checkpoint_1: research 第1轮 → papers_found: [A,B,C,D,E]
checkpoint_2: research 第2轮 → papers_found: [A,B,C,D,E,F,G,H]
checkpoint_3: synthesize 完成 → final_output: "综述..."
```

进程崩溃 → SQLite 文件在硬盘上完好 → 重启用相同 `thread_id` 调用 → 读最后 checkpoint → 从断点继续。

`node_research` 里的 `papers = list(state.get("papers_found", []))` 用 `list()` 创建副本——安全修改副本后通过 `return {"papers_found": papers}` 显式更新，不绕过 LangGraph 的状态追踪。

### 2.4 图拓扑

```
understand → research ⇄ synthesize → user_review → END
入口         迭代搜索    撰写/回答     人机协同
```

3 条路径：
- `/review` → understand → research ↻ → synthesize → user_review → END
- `/research` → understand → research ↻ → synthesize → END（无 user_review）
- `/progress-report` → understand → research ↻ → synthesize → user_review → END

### 2.5 四个节点

**understand**（[graph.py:108-143](research_assistant/agent/graph.py#L108-L143)）：只用 kb + history 工具盘点已有基础 → 产出 `search_plan`。

**research**（[graph.py:150-195](research_assistant/agent/graph.py#L150-L195)）：用全套工具迭代搜索。`papers_found` 跨轮累积。满意度判定双层：硬上限（iteration≥3 或 count≥8）+ 软判断（LLM 输出含"满意""足够"且 count≥3）。

**synthesize**（[graph.py:227-255](research_assistant/agent/graph.py#L227-L255)）：根据 `workflow_type` 选不同 Prompt 模板。temperature=0.5（更创造性），max_tokens=4096。

**user_review**（[graph.py:277-296](research_assistant/agent/graph.py#L277-L296)）：调用 `interrupt(prompt)` 暂停执行 → LangGraph 自动持久化状态 → 用户可关闭程序 → 下次用 `Command(resume=...)` 恢复。

### 2.6 interrupt() 人机协同原理

`interrupt()` 暂停图执行 → SqliteSaver 将完整 state 写入 SQLite → 控制权交还 CLI → CLI 展示 prompt 等用户输入 → 用户输入后调 `graph.stream(Command(resume=user_input))` → LangGraph 从 SQLite 恢复 → `interrupt()` 返回用户输入 → 节点继续执行。

---

## 三、记忆系统

### 3.1 四层架构

| 记忆层 | 存储 | 生命周期 | 核心能力 |
|--------|------|----------|----------|
| 工作记忆 | 内存 | 会话级 | Q&A 多轮上下文，30s 防抖触发事实提取 |
| 用户画像 | `user_facts.json` | 永久 | LLM 提炼的结构化事实（带置信度），始终注入 |
| 情景记忆 | SQLite + Qdrant | 永久 | 会话摘要，混合检索（语义 × 时间衰减） |
| 语义记忆 | Neo4j | 永久 | 实体-关系知识图谱，跨论文推理 |

### 3.2 工作记忆 — WorkingMemory

文件：[memory/working.py](research_assistant/memory/working.py)

核心数据结构：
```python
self.turns: list[QATurn] = []        # 对话轮次（问题+答案+引用+时间戳+信号标记）
self._consolidated: list[str] = []    # 被压缩的历史对话摘要
```

`MAX_TURNS = 10`：超过后最旧的 5 轮被 LLM 压缩为 1-2 句摘要 → 存入 `_consolidated` → 删除旧轮次。

`get_context(n_turns)` 返回 "历史摘要 + 最近 N 轮完整对话" 的格式化文本，注入 QA prompt。

### 3.3 防抖定时器机制

文件：[memory/working.py:166-202](research_assistant/memory/working.py#L166-L202)

```python
def _reset_debounce_timer(self):
    self._debounce_timer.cancel()                    # 取消旧的
    self._debounce_timer = threading.Timer(30.0,     # 新建一次性定时器
                                           self._on_debounce_fire)
    self._debounce_timer.start()
```

每次 `add_turn` 取消旧定时器 → 启动新的 30s 倒计时。用户连续追问时定时器不断重置；用户停止说话 30s 后才触发。

设计目的：
1. **省钱**：批量提取 1 次 LLM 调用 vs 每轮提取 N 次调用
2. **提升质量**：LLM 看到完整对话上下文，提取的事实比逐条提取更准确
3. **非阻塞**：提取在后台线程执行

我们修复了重复传输问题：加了 `_last_extracted_index` 标记，每次提取只传上次提取之后的新轮次。

### 3.4 事实提取管线

文件：[memory/fact_extraction.py](research_assistant/memory/fact_extraction.py)

```
对话 → add_turn → turns 列表 → 30s 防抖 → extract_facts_from_conversation()
  ↓
LLM 输入: 对话文本 + 已有事实摘要 + 纠正/认可信号提示
LLM 输出: {userContext, newFacts: [{content, category, confidence}], factsToRemove}
  ↓
merge_facts(): casefold 去重 + 置信度过滤(<0.5) + 按置信度降序 + 裁剪到100条
  ↓
UserProfileManager.update_facts() → save_facts() → 原子写入 user_facts.json
```

置信度标准：
- 0.95-1.0: 用户明确说出
- 0.70-0.94: 从对话合理推断
- 0.50-0.69: 间接暗示
- <0.50: 不输出

### 3.5 信号检测

文件：[memory/message_processing.py](research_assistant/memory/message_processing.py)

纠正信号（18 个正则模式）："不对"、"应该是"、"你搞错了"、"实际上"...
认可信号（17 个正则模式）："很好"、"谢谢"、"没错"、"完全正确"...

信号跨轮累积（OR 操作），一旦检测到就保持到提取完成。纠正信号触发 `factsToRemove`——LLM 同时输出新事实和要删除的旧事实。

### 3.6 用户画像 — UserProfileManager

文件：[memory/user_profile.py](research_assistant/memory/user_profile.py)

存储：`data/users/{name}/user_facts.json`

```json
{
  "userContext": {"researchFocus": "...", "methodPreference": "...", "expertiseLevel": "..."},
  "facts": [{"content": "...", "category": "experiment_detail", "confidence": 0.95}]
}
```

原子写入：先写临时文件 `xxx.tmp` → `os.replace(tmp, target)`。写到一半崩溃 → 原文件不受影响。

每次 QA 调用 `build_context_for_qa()` → top 15 事实（按置信度排序）→ 注入 System Prompt 的"## 背景上下文"。

### 3.7 情景记忆 — EpisodicMemory

文件：[memory/episodic.py](research_assistant/memory/episodic.py)

写入：会话退出时 LLM 阅读操作日志 → 生成结构化摘要 → 存入 Qdrant（向量可召回）+ SQLite session_log 表。

读出：`recall_context_hybrid()` — 混合检索 = 语义相似度 × 时间衰减：

```python
combined_score = semantic_score × e^(-days_ago / 30)
# 今天: ×1.00  昨天: ×0.97  30天前: ×0.37  60天前: ×0.14
```

### 3.8 语义记忆 — SemanticMemory

文件：[memory/semantic.py](research_assistant/memory/semantic.py) + [memory/knowledge_graph.py](research_assistant/memory/knowledge_graph.py)

触发：论文入库时（ingestion pipeline 第5步），Neo4j 在线时才执行。

流程：LLM 从厚标注中抽取实体（Material/Method/Property/Parameter）和关系（PRODUCES/AFFECTS/HAS_PROPERTY）→ 写入 Neo4j 图 → 关系上标记 paper_id 追溯来源。

图查询能力：
- `get_knowledge_gaps()` — 找关联论文最少的 Property → 知识缺口
- `find_path(from, to)` — 两实体间的关联路径
- `find_contradictions()` — 不同论文对同一实体的矛盾结论
- `suggest_research_direction()` — LLM 基于缺口 + 图统计建议方向

### 3.9 完整闭环

```
┌─ 写路径 ─────────────────────────────────────────┐
│ 对话 → add_turn → 信号检测 → 30s防抖              │
│ → LLM提取结构化事实 → merge_facts → 原子写入       │
│ → 退出时: 会话摘要 → Qdrant + SQLite              │
└───────────────────────────────────────────────────┘

┌─ 读路径 ─────────────────────────────────────────┐
│ QAService._build_context()                        │
│ ├─ 用户画像: user_facts.json (~800t)             │
│ ├─ 工作记忆: turns最近3轮 (~500t)                │
│ ├─ 情景记忆: recall_context_hybrid()             │
│ └─ 未解决问题: session_log (~100t)               │
│ → 拼入 System Prompt → LLM "认识"用户             │
└───────────────────────────────────────────────────┘
```

### 3.10 _consolidated 列表

已压缩的历史对话摘要列表。`turns` 超 10 轮时触发 `_consolidate_oldest()`——取最旧 5 轮 → LLM 压缩为 1-2 句 → 追加到 `_consolidated` → 删除那 5 轮。`get_context()` 先拼摘要再拼最近完整对话，让 LLM 既知道历史主题又能看到最新细节。

### 3.11 已有事实摘要的作用

LLM 提取事实时传入已有事实列表，让 LLM 判断"这条是不是已经知道了"。已有的事实不会重复提取，省 API 成本 + 避免 user_facts.json 膨胀。
