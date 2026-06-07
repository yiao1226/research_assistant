# 科研助手 — 项目全貌

**总代码量**: 45 个 Python 文件，~8,500 行。43 个测试，全部通过。

---

## 一、架构总览

```
main.py (209行)                 ← CLI 入口
  │
cli/
  ├─ commands.py (600+行)       ← 所有命令实现
  └─ display.py (42行)          ← Banner + 帮助文本
  │
research_assistant/
  ├─ loaders/                   ← 文档加载（PDF + 非PDF）
  │   ├─ loader.py              # 统一入口，按文件类型分发
  │   ├─ pymupdf_loader.py      # PDF 主力：伪MD注入 + 页眉页脚清洗
  │   ├─ markitdown_loader.py   # 非PDF 格式 (Word/Excel/PPT/图片)
  │   └─ metadata.py            # 标题/摘要/论文类型检测
  │
  ├─ rag/                       ← RAG 管道（文档→检索→回答）
  │   ├─ embedding.py           # BGE 嵌入模型封装 (512维)
  │   ├─ vector_store.py        # Qdrant 向量存储单例封装
  │   ├─ chunking.py            # 句子窗口分块 + CJK Token估算
  │   ├─ retrieval.py           # 混合检索：BM25 + Dense + RRF + LLM re-rank
  │   └─ ingestion.py           # 入库管线：LLM厚标注 → 分块 → 嵌入 → 双写
  │
  ├─ core/                      ← 基础设施
  │   ├─ storage.py             # SQLite CRUD（5表 + FTS5 全文索引）
  │   ├─ user_manager.py        # 多用户文件系统隔离
  │   └─ backup.py              # 4层备份 + 操作日志（含丰富摘要）
  │
  ├─ memory/                    ← 记忆系统（多层，参考 DeerFlow 架构）
  │   ├─ working.py             # 工作记忆：会话级上下文 + 防抖定时器
  │   ├─ episodic.py            # 情景记忆：会话摘要 + 混合检索（语义×时间）
  │   ├─ semantic.py            # 语义记忆：Neo4j 知识图谱
  │   ├─ knowledge_graph.py     # Neo4j CRUD 底层
  │   ├─ user_profile.py        # 用户画像：结构化事实持久化（user_facts.json）
  │   ├─ fact_extraction.py     # 事实提取：LLM提取 + 置信度 + 增量合并
  │   ├─ message_processing.py  # 信号检测：纠正/认可信号正则匹配
  │   └─ debounce_queue.py      # 防抖队列：多用户记忆更新队列
  │
  ├─ tools/                     ← 业务工具
  │   ├─ qa.py                  # 智能问答：4意图（chat/recall/research/direction）
  │   ├─ search_orchestrator.py # 搜索编排：画像 → 扩展 → 多源搜索 → 排序
  │   ├─ paper_search.py        # ArXiv/Semantic Scholar/WoS API 调用
  │   ├─ upload.py              # PDF 上传 + 匹配卡片 + 入库
  │   ├─ progress.py            # 进展记录 + 摘要查看
  │   └─ plan.py                # 研究计划管理
  │
  ├─ graph.py                   # LangGraph 工作流（9节点文献综述）
  ├─ state.py                   # 状态定义 + RuntimeContext
  ├─ prompts.py                 # Agent 提示词模板
  │
  └─ utils/                     ← 基础工具
      ├─ llm_factory.py         # LLM 工厂（全项目唯一入口）
      └─ json_utils.py          # LLM 响应 JSON 解析
```

---

## 二、功能清单

### 1. 文档加载与解析

**入口**: `upload` 命令 → `cli/commands.py:run_upload()` → `tools/upload.py:UploadManager`

**流程**:
```
PDF 文件 → loaders/loader.py:DocumentLoader.load()
  ├─ .pdf → PyMuPDF 提取 → 行清洗 → 伪MD注入 (pymupdf_loader.py)
  │         页面标题/摘要/类型检测 (metadata.py)
  │         页眉页脚频率统计剔除
  │         章节号正则匹配注入 ### 标题
  │
  ├─ .docx/.xlsx/.pptx/.html/.jpg → MarkItDown 转换 (markitdown_loader.py)
  │
  └─ 纯文本 → 直接读取
```

**关键文件**: `loaders/loader.py`, `loaders/pymupdf_loader.py`, `loaders/metadata.py`

---

### 2. 论文搜索与个性化推荐

**入口**: `search` 命令 → `tools/search_orchestrator.py:SearchOrchestrator`

**流程**:
```
用户输入关键词
  → LLM 构建用户研究画像 (进展 + 已有论文方向 + 知识缺口)
  → LLM 扩展查询词 (多角度关键词)
  → 多源并行搜索 (ArXiv + Semantic Scholar + Web of Science)
  → 多因素排序: 语义匹配(40%) + 关键词(25%) + 热门度(20%) + 时效(15%)
  → LLM 生成 Top-5 排名理由
  → 用户选择入库 / 下载 / 跳过
```

**关键文件**: `tools/search_orchestrator.py`, `tools/paper_search.py`

---

### 3. 入库管线

**流程**:
```
论文 dict + 全文 Markdown
  → LLM 厚标注 (~800-2600 token 一次性标注)
     材料关键词 / 方法关键词 / 现象关键词
     方法详情 (参数、设备、测量什么)
     关键发现 / 核心结论 / 遗留缺口
  → 标注完整性验证 + 不足二次补全
  → 句子窗口分块 (chunking.py)
     索引粒度: 单个句子
     上下文窗口: 前后各3句
  → BGE 嵌入 (512维) → Qdrant 向量
  → 元数据 + 标注 → SQLite
  → Neo4j 语义记忆提取 (实体-关系图谱)
```

**关键文件**: `rag/ingestion.py`, `rag/chunking.py`, `memory/semantic.py`

---

### 4. 多层记忆系统（参考 DeerFlow 架构改进）

记忆系统经 P0+P1 改进后，形成完整的写入→存储→检索→注入闭环。

#### 4.1 记忆分层

| 记忆层 | 存储 | 生命周期 | 核心能力 |
|--------|------|----------|----------|
| **工作记忆** | 内存 | 会话级 | Q&A 多轮对话上下文，防抖触发事实提取 |
| **用户画像** | `user_facts.json` | 永久 | LLM 提炼的结构化事实（带置信度），始终注入 |
| **情景记忆** | SQLite + Qdrant | 永久 | 会话摘要 + 操作日志，混合检索（语义×时间） |
| **语义记忆** | Neo4j 图数据库 | 永久 | 实体-关系知识图谱，跨论文推理 |

#### 4.2 写入路径（对话 → 事实）

```
add_turn()
  ├─ ① 信号检测（message_processing.py）
  │     classify_turn(question) → correction_detected / reinforcement_detected
  │
  ├─ ② 追加到 turns 列表
  │
  └─ ③ 防抖定时器（30秒）
        _reset_debounce_timer() → 每次新轮次取消旧定时器，新建倒计时
        │
        └─ 定时器到期 / 退出时 add_nowait_flush()
             │
             ├─ 读已有事实（user_facts.json）
             ├─ LLM 提取新事实（fact_extraction.py）
             │   提取 {content, category, confidence} 三元组
             │   纠正信号 → 置信度 >= 0.90
             │
             ├─ merge_facts() 增量合并
             │   · 同内容覆盖（casefold 去重）
             │   · 置信度 < 0.5 过滤
             │   · 超过 100 条裁剪
             │
             └─ 原子写入 user_facts.json（temp file + os.replace）
```

#### 4.3 读取路径（问答时注入）

```
qa.py ask() → _build_intent_context()
  │
  ├─ ① 用户画像（始终注入，~800 tokens）
  │     user_facts.json → 研究方向 + top 15 事实
  │
  ├─ ② 工作记忆（始终注入，~500 tokens）
  │     self.turns → 最近 3 轮对话
  │
  ├─ ③ 情景记忆（按意图触发）
  │     recall intent → 必须检索
  │     direction intent → 必须检索
  │     research intent → 论文 < 3 篇时检索
  │     chat intent → 不检索
  │     recall_context_hybrid() → 语义相似度 × e^(-days/30)
  │
  └─ ④ 未解决问题（始终注入，~100 tokens）
        SQLite session_log → 最近 3 条
```

#### 4.4 会话退出时

```
_on_quit()
  ├─ add_nowait_flush()          ← 不等定时器，立即提取事实
  ├─ end_session()               ← 保存工作记忆到情景记忆
  └─ generate_session_summary()  ← LLM 读操作日志生成摘要 → Qdrant + SQLite
```

#### 4.5 操作日志（丰富摘要）

入库/综述/进展操作记录含核心内容，不再只有标题：
- `log_upload`: 论文标题 + 核心结论/摘要
- `log_review`: 综述主题 + 综述结论（首300字）
- `log_progress`: 进展标题 + 描述 + 结果 + 洞察

---

### 5. 智能问答 (Q&A) — 4意图架构

**入口**: `ask` 命令 → `tools/qa.py:QAService`

**4 种意图**:

| 意图 | 触发条件 | 检索范围 | LLM 调用次数 |
|------|----------|----------|------------|
| `chat` | 问候、概念解释 | 不检索 | 1次（Phase1内回答） |
| `recall` | 询问历史讨论 | 仅情景记忆 | 2次 |
| `research` | 需要查论文的科研问题 | 论文+进展+图谱+情景记忆(论文少时) | 2次 |
| `direction` | research + 想要后续方向 | 全部（含情景记忆） | 3次（含方向提取） |

**Phase 1 注入内容**: 用户画像（800t）+ 工作记忆（500t）+ 未解决问题（100t）+ 进展摘要（200t）≈ 1,850 tokens
**Phase 2 注入内容**: 论文（~1,700t）+ 进展 + 图谱 + 情景记忆 + 工作记忆 ≈ 2,900 tokens

---

### 6. 混合检索 (MQE + HyDE 扩展)

**入口**: `rag/retrieval.py:HybridRetriever.search_papers_expanded()`

**查询扩展（检索前）**:
```
search_papers_expanded()
  ├─ MQE (Multi-Query Expansion)
  │   LLM 生成 3 个语义等价的多样化查询
  │   "CVD温度优化" → ["MPCVD沉积温度参数优化",
  │                     "金刚石薄膜生长温度影响",
  │                     "CVD process temperature optimization"]
  │
  └─ HyDE (Hypothetical Document Embeddings)
      始终执行。LLM 生成假设性答案段落 → 嵌入 → 用假设答案搜知识库
      上下文来源（优先级从高到低）:
        1. 论文厚标注关键词（材料/方法/现象）
        2. 用户画像结构化事实（user_facts.json top 5）
        3. 研究进展记录
      上下文越丰富 → 假设答案与知识库用词越一致 → 检索越精准
      无上下文时仍可用，LLM 通用知识生成的段落仍比原始问题更像论文文本
```

**四路融合**:
```
1. BM25 (论文元数据关键词，jieba 中文分词)
2. Dense (Qdrant 语义向量 — 句子级索引)
     → 后处理: window_text 替换 → 宽上下文
3. SQLite (FTS5 全文索引)
4. RRF 融合 → heading_path 关键词加权 → LLM re-rank → top-k
```

**句子窗口策略** (`rag/chunking.py`):
```
索引: 单个句子 → 高精度检索
窗口: 前后各3句 → 丰富上下文给 LLM
heading_path: 携带章节路径 → 检索结果可定位
命中标记: ▶ 精确显示匹配句
```

**关键文件**: `rag/retrieval.py`, `rag/chunking.py`, `rag/vector_store.py`

---

### 7. 研究进展追踪

**入口**: `record` / `progress` 命令 → `tools/progress.py`

**关键文件**: `tools/progress.py`, `tools/plan.py`

---

### 8. 文献综述工作流 (LangGraph)

**入口**: `review` 命令 → `graph.py`

**9 节点工作流**:
```
understand_intent → search_papers(三路合流) → analyze_papers(分层)
  → synthesize_review → plan_research → user_progress_input(中断点)
  → assess_progress → suggest_next → END
```

**关键文件**: `graph.py`, `state.py`, `prompts.py`

---

### 9. 跨会话回忆

已集成到 QA 智能问答中，无需手动命令：

- QA `recall` 意图 → `recall_context_hybrid()` 混合检索（语义相似度 × 时间衰减）
- 用户问"上次讨论了什么"时，LLM 自动识别为 `recall`，搜索情景记忆后回答

---

### 10. 多用户隔离

每个用户独立数据空间:
```
data/users/{username}/
  ├─ library.db          (SQLite 论文/进展/计划/搜索历史/操作日志)
  ├─ user_facts.json     (用户画像：结构化事实)
  ├─ checkpoints.db      (LangGraph 状态持久化)
  ├─ backups/            (JSON 快照)
  ├─ inbox/              (PDF 拖拽上传)
  └─ logs/               (分类操作日志)
Qdrant collection: user_{username}_papers/_progress/_memory
```

---

## 三、数据流总图 

```
外部论文 API (ArXiv/S2/WoS)
  │
  ▼
search_orchestrator.py    ← LLM画像 + 多因素排序
  │
  ▼ (用户选择入库)
ingestion.py              ← LLM厚标注
  ├─ chunking.py          ← 句子窗口分块
  ├─ vector_store.py      ← BGE嵌入 → Qdrant
  ├─ storage.py           ← SQLite双写
  └─ semantic.py          ← Neo4j知识图谱
  │
  ▼ (用户提问)
qa.py                     ← 4意图识别
  │
  ├─ intent=recall ──→ _retrieve_episodic() → 情景记忆混合检索
  ├─ intent=chat ────→ 直接回答（不检索）
  └─ intent=research/direction
       ├─ retrieval.py    ← BM25 + Dense + RRF + heading_path boost
       ├─ episodic.py     ← recall_context_hybrid()（语义×时间）
       └─ semantic.py     ← Neo4j 知识图谱
  │
  ▼ (回答后)
working.py                ← add_turn()
  ├─ message_processing   ← 信号检测
  └─ _reset_debounce_timer() ← 30秒防抖
       │
       ▼ (定时器到期/退出)
  _extract_and_persist_facts()
       ├─ fact_extraction  ← LLM提取结构化事实
       ├─ merge_facts()    ← 置信度过滤+去重+裁剪
       └─ user_profile.py  ← 原子写入 user_facts.json
  │
  ▼ (退出时)
episodic.py               ← 会话摘要 → Qdrant + SQLite
backup.py                 ← 丰富操作日志 → 分类日志文件
```

---

## 四、环境依赖

```
Python 3.10 | Conda: research_agent
Docker: Qdrant (localhost:6334) + Neo4j (localhost:7687)
LLM: DeepSeek API (deepseek-v4-flash)
嵌入: BAAI/bge-small-zh-v1.5 (512维, 96MB)
```

**启动**:
```bash
conda activate research_agent
docker start qdrant neo4j
cd C:\Users\yiyao1226\Desktop\Project2
python main.py
```

**CLI 命令**:
```
ask <问题>                       智能问答（4意图自动路由，含历史回忆）
search [-s arxiv|s2|wos] 关键词  个性化论文搜索
upload <pdf>                      上传论文
download <arxiv_id>               下载论文
review <主题>                     文献综述
record <主题>                     记录进展
progress <主题>                   查看进展
backup                            备份数据
user list | switch | delete       用户管理
```

---

## 五、新增模块速查（P0+P1 改进）

| 文件 | 行数 | 功能 | 测试覆盖 |
|---|---|---|---|
| `memory/message_processing.py` | 126 | 纠正/认可信号正则检测 | 9 |
| `memory/debounce_queue.py` | 251 | 防抖记忆更新队列 | 8 |
| `memory/fact_extraction.py` | 382 | LLM结构化事实提取+置信度合并 | 10 |
| `memory/user_profile.py` | 183 | 用户画像原子读写+被动注入 | 7 |
| `tests/test_memory_v2.py` | 392 | 43个纯函数测试（零外部依赖） | — |
| `tests/demo_memory_v2.py` | 220 | 直白演示脚本（5个演示） | — |

**修改的文件**: `memory/working.py`, `tools/qa.py`, `memory/episodic.py`, `core/backup.py`, `cli/commands.py`, `memory/__init__.py`
