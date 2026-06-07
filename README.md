# 科研助手 — Research Assistant

AI 驱动的科研工作平台。Agent 自主决策 + LangGraph 深度工作流 + 多层记忆系统。

## 快速开始

```bash
# 环境
conda activate research_agent        # Python 3.10
docker start qdrant neo4j            # 向量库 + 图数据库

# 启动
cd yiao1226_research_assistant
python main.py
```

```
==================================================
  科研助手 - Research Assistant
  LangChain + LangGraph + Qdrant + Neo4j
  智能问答 | 语义检索 | 三层记忆 | 个性化分析
==================================================

直接输入问题即可 ── Agent 自动分析 + 工具调用
  /search 或 /s    关键词 — 外部论文搜索
  /review 或 /r    主题   — 文献综述 (Agent驱动)
  /research 或 /rs 主题   — 深度研究分析
  /upload 或 /u    <pdf>  — 上传论文
  /record 或 /n    主题   — 记录进展
  /progress 或 /p  主题   — 查看进展
  /backup                  — 备份数据
  help                     — 帮助
  quit                     — 退出

[LAB] yiao1226 >
```

## 使用示例

### 日常科研问答（默认 Agent 模式）

直接说话，Agent 自主决定搜索策略：

```
[LAB] yiao1226 > 金刚石CVD温度600度和400度哪个好

🤖 分析: 金刚石CVD温度600度和400度哪个好...
  📚 知识库检索 query_knowledge_base({'query': 'CVD 温度 优化 金刚石'})
     → [论文1] 金刚石薄膜CVD制备研究 核心: 600℃沉积Ra 2.1nm...

根据你的知识库论文[1]和实验记录，600℃沉积温度下表面粗糙度Ra 2.1nm，
显著优于400℃的Ra 3.4nm...

📚 引用论文: 1篇
```

### 外部论文搜索（带交互选择）

```
[LAB] yiao1226 > 帮我搜钙钛矿LED最新进展

🤖 分析: 帮我搜钙钛矿LED最新进展...
  🔍 外部搜索 search_papers_online({'query': 'perovskite LED'})
     ⏳ 搜索 ArXiv + Semantic Scholar...

  ───────────────────────────────────────────────────────
  🔍 找到 15 篇, 精选 Top-5 | 耗时 8.3s
  ───────────────────────────────────────────────────────

  [1] Efficient Perovskite QD LEDs via Ligand Engineering
      92/100 | 引用: 147 | 2025
      配体交换将PLQY从72%提升至95%

  [2] Stable Blue Perovskite Quantum Dot LEDs
      85/100 | 引用: 89 | 2025
      通过核壳结构实现蓝色QLED稳定性突破

  ...

  [I] 入库 (如 I1,3)  [D] 下载 (如 D1)  [S] 跳过  [Q] 追问
  > I1,2

     ✅ 已入库 [1]: Efficient Perovskite QD LEDs... (ID=44)
     ✅ 已入库 [2]: Stable Blue Perovskite... (ID=45)
```

### 文献综述（LangGraph 深度工作流）

```
[LAB] yiao1226 > /review 金刚石薄膜CVD制备

[START] 启动文献综述: 金刚石薄膜CVD制备

──────────────────────────────────────────────────
  📊 盘点现状 + 制定策略
──────────────────────────────────────────────────
  KB有5篇金刚石论文, 进展有2条CVD实验记录
  搜索计划: diamond CVD temperature optimization, MPCVD substrate...

──────────────────────────────────────────────────
  🔍 搜索分析
──────────────────────────────────────────────────
  第1轮 | 累计: 8篇 | 🔄 继续
  第2轮 | 累计: 13篇 | ✅ 满意

──────────────────────────────────────────────────
  📝 综合撰写
──────────────────────────────────────────────────
  产出: 3800字 | 引用: 13篇
```

### 文件拖入

```
[LAB] yiao1226 > C:\论文\钙钛矿量子点综述.pdf

📄 正在分析: 钙钛矿量子点综述.pdf
   解析完成: 142页, 85000字符 (pymupdf_fallback)

───────────────────────────────────────────────────────
  标题: 钙钛矿量子点发光二极管的研究进展
  分析: 综述了钙钛矿QLED的发光机理、制备工艺和PLQY提升策略...
───────────────────────────────────────────────────────

  [I] 入库到知识库  [Q] 追问内容  [S] 跳过
```

## 核心能力

### 智能问答（qa.py）

Agent + 5 工具自主调用，不固定意图路由：

| 工具 | 用途 |
|---|---|
| `query_knowledge_base` | 本地论文语义检索（BM25 + Dense + RRF） |
| `query_progress` | 用户研究进展查询 |
| `recall_history` | 历史会话回忆（语义 × 时间衰减） |
| `search_papers_online` | 外部论文搜索（ArXiv + S2，带交互入库） |
| `ingest_papers` | 入库最近搜索结果 |

### LangGraph 工作流（agent/graph.py）

4 节点 / 3 路径 / 有状态 / 可迭代：

```
/review          → understand → research(迭代) → synthesize → user_review
/research        → understand → research(迭代) → synthesize → END
/progress-report → understand → research(迭代) → synthesize → user_review
```

- **understand**: Agent 盘点 KB/进展/历史 → 制定搜索策略
- **research**: 搜索分析循环，不满意换策略重搜（最多 3 轮）
- **synthesize**: 按路径产出综述/回答/评估报告
- **user_review**: `interrupt()` 人机协同，可修改后重写

### 文档解析（loaders/）

- PDF: PyMuPDF 主力，伪 Markdown 注入 + 页眉页脚清洗
- 非 PDF: MarkItDown（Word/Excel/PPT/图片）

### 混合检索（retrieval/）

- BM25（jieba 中文分词）+ Dense（BGE 512 维）+ RRF 融合
- MQE 多查询扩展 + HyDE 假设文档
- 句子窗口分块（索引粒度单句，上下文窗口前后 3 句）

### 多层记忆（memory/）

| 记忆层 | 存储 | 生命周期 |
|---|---|---|
| 工作记忆 | 内存 | 会话级 |
| 用户画像 | `user_facts.json` | 永久 |
| 情景记忆 | SQLite + Qdrant | 永久 |
| 语义记忆 | Neo4j 图数据库 | 永久 |

- 30 秒防抖队列 → LLM 提取结构化事实
- 纠正/认可信号检测
- 原子写入 + 置信度过滤 + Top-100 裁剪

## 架构

```
main.py                     ← CLI 入口（Agent / 命令 / 文件检测）
  │
research_assistant/
  ├── agent/                ← Agent 编排层
  │   ├── qa.py             ← 智能问答（Agent + Hook + 流式）
  │   ├── graph.py          ← LangGraph 工作流（4节点/3路径）
  │   ├── state.py          ← 工作流状态
  │   └── prompts.py        ← 提示词模板
  │
  ├── tools/                ← 工具层（Agent + Graph 共用）
  │   ├── kb.py             ← query_knowledge_base + query_progress
  │   ├── search.py         ← search_papers_online + ingest_papers
  │   ├── history.py        ← recall_history
  │   ├── search_orchestrator.py ← 搜索编排
  │   ├── paper_search.py   ← ArXiv/S2/WoS API
  │   ├── upload.py         ← 论文上传
  │   └── progress.py       ← 进展记录
  │
  ├── retrieval/            ← RAG 管道
  │   ├── hybrid.py         ← 混合检索器
  │   ├── chunking.py       ← 句子窗口分块
  │   ├── embedding.py      ← BGE 嵌入
  │   ├── vector_store.py   ← Qdrant 封装
  │   └── ingestion.py      ← 入库管线
  │
  ├── memory/               ← 记忆系统
  │   ├── working.py        ← 工作记忆
  │   ├── episodic.py       ← 情景记忆
  │   ├── semantic.py       ← 语义记忆（Neo4j）
  │   ├── profile.py        ← 用户画像
  │   ├── facts.py          ← 事实提取
  │   ├── signals.py        ← 信号检测
  │   └── queue.py          ← 防抖队列
  │
  └── loaders/              ← 文档加载
      ├── loader.py         ← 统一入口
      ├── pdf.py            ← PyMuPDF
      ├── formats.py        ← MarkItDown
      └── metadata.py       ← 元数据提取
```

## 技术栈

| 组件 | 技术 |
|---|---|
| LLM | DeepSeek V4 |
| Agent | LangChain + 自定义循环 + Hook |
| 工作流 | LangGraph + SqliteSaver |
| 向量库 | Qdrant (BGE-small-zh-v1.5, 512维) |
| 图库 | Neo4j |
| 存储 | SQLite + FTS5 全文索引 |
| 测试 | pytest, 43 个（零外部依赖） |

## 配置

复制 `.env.example` → `.env`，填入 API Key：

```bash
LLM_MODEL_ID=deepseek-v4-flash
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=tvly-xxx
QDRANT_URL=http://localhost:6334
NEO4J_URL=bolt://localhost:7687
```

## 开发

```bash
# 测试
pytest tests/ -v

# 只跑 QA 测试（零外部依赖）
pytest tests/test_qa.py -v

# 只跑记忆测试
pytest tests/test_memory_v2.py -v
```
