# ADR-002: 四层记忆系统架构

**状态**: 已采纳  
**日期**: 2025-06  
**决策者**: @yiyao1226

---

## Context（背景）

参考 DeerFlow（HelloAgents 开源记忆系统）的分层设计理念。
在 RAG Agent 中，仅仅把聊天历史塞进 prompt 是不够的——

1. **对话内容有多样性**：用户的一句话可能包含研究偏好（"我主要做 CVD"）、
   实验参数（"温度 850°C"）、纠正信号（"不对，那个是 750°C"）。
   不同内容需要不同的存储和处理策略。

2. **生命周期不同**：当前会话的上下文只在这 10 分钟内有用；
   研究偏好需要记住几个月；跨论文的知识关联需要永久存储。

3. **检索模式不同**：工作记忆是 FIFO（最新优先）；情景记忆需要语义 × 时间衰减；
   语义记忆需要图遍历（"这种表征方法还在哪些材料上用过？"）。

## Decision（决策）

采用四层记忆架构，每层有不同的存储介质、生命周期和检索方式：

| 记忆层 | 存储 | 生命周期 | 容量 | 核心操作 |
|--------|------|----------|------|----------|
| **工作记忆** (Working) | 内存 list | 会话级 | 10 轮 + 压缩 | `get_context(n)` 最近 N 轮 |
| **用户画像** (User Profile) | JSON 文件 | 永久 | ~100 条事实 | `build_context_for_qa()` 注入 prompt |
| **情景记忆** (Episodic) | SQLite + Qdrant | 永久 | 按天聚合 | `recall_context_hybrid()` 语义 × e^(-d/30) |
| **语义记忆** (Semantic) | Neo4j 图数据库 | 永久 | 不限 | `extract_and_store()` 实体-关系图谱 |

### 为什么不同层用不同存储

```
工作记忆 ──→ 内存 ──→ 原因: 极低延迟（<1ms），不需要持久化，退出就过期

用户画像 ──→ JSON ──→ 原因: ~100 条结构化事实，LLM 提取一次只需几 KB
                        原子写入（temp + os.replace），不需要事务
                        过度工程化用 DB 反而增加延迟

情景记忆 ──→ SQLite + Qdrant ──→ 原因: 需要结构化字段（日期、类型）+
                                   语义向量（相似会话检索）
                                   时间衰减要求 score × e^(-days/30)

语义记忆 ──→ Neo4j ──→ 原因: "方法 M 用在材料 A 和 B 上" → 多跳遍历
                            Neo4j 的 Cypher 查询天然适合这种模式
                            向量 DB 做这个需要多次 round-trip
```

### 写入路径（对话 → 事实提取）

```
add_turn(question, answer)
  │
  ├─ ① 信号检测（正则匹配 "不对"/"应该是"/"没错"）
  │     correction_detected / reinforcement_detected
  │
  ├─ ② 追加到 turns 列表
  │
  └─ ③ 防抖定时器重置（30 秒）
        │
        └─ 到期 → LLM 提取结构化事实
             │
             ├─ 只传新轮次（_last_extracted_index 追踪）
             ├─ 已有事实做去重参考
             ├─ 纠正信号 → 置信度 ≥ 0.90
             ├─ merge_facts() 增量合并
             └─ 原子写入 user_facts.json
```

### 读取路径（问答时注入 prompt）

```
QAService._build_context()
  │
  ├─ ① 用户画像（始终注入，~800 tokens）
  ├─ ② 工作记忆（始终注入，最近 3 轮，~500 tokens）
  ├─ ③ 情景记忆（按意图触发，~500 tokens）
  │     recall/direction → 必须检索
  │     research → 论文 < 3 篇时检索
  │     chat → 不检索（闲聊不需要历史）
  └─ ④ 未解决问题（始终注入，~100 tokens）
```

## Consequences（后果）

### 正面
- 记忆写入不阻塞对话 — 防抖定时器避免每轮都调 LLM
- 退出时 `add_nowait_flush()` 保证不丢数据
- 四层各司其职，不会被单一 DB 的性能瓶颈拖累
- 时间衰减 `e^(-days/30)` 让旧记忆自然"淡出"

### 负面
- 四层记忆维护成本高 — 每层的索引/清理逻辑独立
- Neo4j 增加了部署复杂度（需要 Docker 容器）
- 事实提取的质量依赖 LLM 能力和 prompt（置信度阈值是经验值）

### 风险
- 防抖定时器在退出时如果忘记 flush，最后一轮对话可能丢失
- `_last_extracted_index` 在 turns 压缩后需要同步调整（已修复）

## Alternatives Considered

### A. 全部用 Qdrant 一个 collection
- **否决原因**：不同记忆的检索模式完全不同。
  工作记忆是 FIFO，情景记忆是语义 × 时间，语义记忆是图遍历。
  强行塞进向量 DB 会让所有检索退化到 cosine similarity。

### B. 用 Redis 做工作记忆
- **否决原因**：个人单机场景不需要 Redis。内存 list + 压缩足够了。
  但如果做多实例部署，Redis 是正确的选择。

### C. 不用 Neo4j，用 Qdrant + metadata filter 模拟实体关系
- **否决原因**：metadata filter 可以查 "材料=金刚石"，
  但做不了 "材料 → 方法 → 性能指标" 的两跳推理。
  图数据库的 edge traversal 在这个场景下是本质优势。

---

## 面试要点

- **核心概念**：不是所有记忆都该存向量 DB — 读写模式决定存储选择
- **关键设计**：防抖定时器（避免频繁 LLM 调用）+ 原子写入（防止数据损坏）
- **trade-off**：四层维护成本 vs 检索质量 — 单人项目可控，多实例需简化
- **工程细节**：`_last_extracted_index` 的同步问题及修复
