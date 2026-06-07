"""CLI 命令执行 — 所有 run_* 函数和辅助函数。"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from research_assistant.core import UserManager, BackupManager

# 延迟导入：BGE 模型 96MB + Neo4j，用到时才加载
_EpisodicMemory = None
_VectorStore = None
_HybridRetriever = None
_IngestionPipeline = None


def _get_em():
    global _EpisodicMemory
    if _EpisodicMemory is None:
        from research_assistant.memory import EpisodicMemory
        _EpisodicMemory = EpisodicMemory
    return _EpisodicMemory
_VectorStore = None
_HybridRetriever = None
_IngestionPipeline = None


def _get_vs():
    global _VectorStore
    if _VectorStore is None:
        from research_assistant.rag import VectorStore
        _VectorStore = VectorStore
    return _VectorStore


def _get_hr():
    global _HybridRetriever
    if _HybridRetriever is None:
        from research_assistant.rag import HybridRetriever
        _HybridRetriever = HybridRetriever
    return _HybridRetriever


def _get_ip():
    global _IngestionPipeline
    if _IngestionPipeline is None:
        from research_assistant.rag import IngestionPipeline
        _IngestionPipeline = IngestionPipeline
    return _IngestionPipeline
from research_assistant.tools import (
    search_arxiv, search_semantic_scholar, search_web_of_science,
    download_paper, fetch_paper_full_text,
)
# 重型导入延迟：progress/plan/orchestrator/uploader 都触发 langchain_openai ~7s
from research_assistant.state import set_runtime_context

# 延迟导入：plan.py 导入 langchain_openai ~7s，只有 record 命令需要
_plan_funcs = None


def _get_plan_funcs():
    global _plan_funcs
    if _plan_funcs is None:
        from research_assistant.tools.plan import update_plan_from_progress, detect_new_directions
        _plan_funcs = (update_plan_from_progress, detect_new_directions)
    return _plan_funcs

# 延迟导入：LangGraph 12s，只有 review 命令需要
_build_research_graph = None


def _get_graph():
    global _build_research_graph
    if _build_research_graph is None:
        from research_assistant.graph import build_research_graph as brg
        _build_research_graph = brg
    return _build_research_graph

from .display import print_banner

logger = logging.getLogger(__name__)

# ── 全局引用（由 main.py 注入）──
user_manager: UserManager = None
BASE_DIR = None
AVAILABLE_SOURCES = {}
TYPE_LABELS = {}

_qa_instance = None
_storage_cache = None


def get_storage():
    u = user_manager.current_user
    if u:
        u.init_storage()
        return u.storage
    return None


def get_orchestrator():
    s = get_storage()
    if s:
        from research_assistant.tools.search_orchestrator import SearchOrchestrator
        return SearchOrchestrator(s, user_manager.current_user.username)
    return None


def get_uploader():
    s = get_storage()
    if s:
        from research_assistant.tools.upload import UploadManager
        return UploadManager(s, user_manager.current_user.username)
    return None


def get_episodic():
    s = get_storage()
    if s:
        return _get_em()(s, user_manager.current_user.username)
    return None


def get_backup():
    s = get_storage()
    if s:
        return BackupManager(s, user_manager.current_user.username)
    return None


def get_qa():
    from research_assistant.tools.qa import QAService
    global _qa_instance
    if not user_manager.is_logged_in:
        return None
    s = get_storage()
    if not s:
        return None
    if _qa_instance is None:
        _qa_instance = QAService(user_manager.current_user.username, s)
    return _qa_instance


# ── 命令实现 ──

def parse_search_args(arg):
    parts = arg.split()
    sources = list(AVAILABLE_SOURCES.keys())
    keywords_start = 0
    if parts and parts[0] == "-s":
        if len(parts) < 3:
            print("用法: search [-s arxiv|s2|wos] 关键词1 [关键词2 ...]")
            return [], ""
        source_key = parts[1].lower()
        if source_key not in AVAILABLE_SOURCES:
            print(f"未知搜索源: {source_key}。可用: {', '.join(AVAILABLE_SOURCES.keys())}")
            return [], ""
        sources = [source_key]
        keywords_start = 2
    keywords = parts[keywords_start:]
    if not keywords:
        return [], ""
    query = " AND ".join(keywords)
    return sources, query


def run_search(arg):
    sources, query = parse_search_args(arg)
    if not sources or not query:
        return

    orch = get_orchestrator()
    if not orch:
        print("请先登录用户。")
        return

    print(f"\n[SEARCH] 关键词: {query.replace(' AND ', ', ')}")
    print(f"[SEARCH] 搜索源: {', '.join(sources)}")
    print(f"[SEARCH] 分析用户研究画像 + 扩展查询...")

    result = orch.search(query, sources=sources)
    papers = result.get("papers", [])

    print(f"\n找到 {result.get('total_found', 0)} 篇论文，精选 Top-{len(papers)}")
    print(f"搜索策略: {result.get('search_focus', '')}")
    print(f"耗时: {result.get('duration_sec', 0):.1f}s\n")

    for i, p in enumerate(papers, 1):
        title = p.get("title", "(无标题)")
        score = p.get("composite_score", 0)
        src = p.get("source", "?")
        citations = p.get("citation_count", 0) or 0
        year = p.get("year", "") or ""

        print(f"{'='*60}")
        print(f"  {i}. {title}")
        print(f"{'='*60}")
        print(f"  综合评分: {score}/100 | 引用: {citations} | 年份: {year} | 来源: {src}")
        for key, label in [("core_contribution", "核心贡献"),
                            ("innovation", "创新点"),
                            ("method_brief", "方法"),
                            ("relevance_reason", "与你的相关性"),
                            ("ranking_reason", "排名理由")]:
            val = p.get(key, "")
            if val:
                print(f"  {label}: {val}")
        arxiv_id = p.get("arxiv_id", "")
        if arxiv_id:
            print(f"  ArXiv ID: {arxiv_id}")
        abstract = p.get("abstract", "")
        if abstract:
            print(f"  原文摘要: {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
        print()

    if papers:
        print(f"[TIP] 共找到 {result.get('total_found', 0)} 篇论文，展示 Top-{len(papers)}")
        print("  操作: 输入编号入库 (1,3,5) | d+编号下载 PDF (d1,d3) | all 全部入库 | none 跳过")
        choice = input("  > ").strip()
        _handle_search_choice(choice, papers, result)


def _handle_search_choice(choice, papers, result):
    if choice.lower() == "none":
        return
    ingest_nums, download_nums = [], []
    if choice.lower() == "all":
        ingest_nums = list(range(1, len(papers) + 1))
    else:
        for part in choice.split(","):
            part = part.strip()
            if part.lower().startswith("d"):
                try:
                    n = int(part[1:])
                    if 1 <= n <= len(papers):
                        download_nums.append(n)
                except ValueError:
                    pass
            else:
                try:
                    n = int(part)
                    if 1 <= n <= len(papers):
                        ingest_nums.append(n)
                except ValueError:
                    pass

    if ingest_nums:
        storage = get_storage()
        pipeline = _get_ip()(storage, user_manager.current_user.username)
        for idx in ingest_nums:
            p = papers[idx - 1]
            try:
                pid = pipeline.ingest(p)
                print(f"  已入库: {p.get('title', '')[:60]} (ID={pid})")
            except Exception:
                logger.warning(f"入库失败 [{idx}]", exc_info=True)
                print(f"  入库失败 [{idx}]: {p.get('title', '')[:60]}")
        print(f"共入库 {len(ingest_nums)} 篇论文")

    if download_nums:
        for idx in download_nums:
            p = papers[idx - 1]
            arxiv_id = p.get("arxiv_id", "")
            if not arxiv_id:
                print(f"  [{idx}] 无 ArXiv ID，无法下载")
                continue
            r = download_paper.invoke({"arxiv_id": arxiv_id})
            try:
                data = json.loads(r) if isinstance(r, str) else r
                print(f"  [{idx}] {data.get('message', r)}")
            except Exception:
                print(f"  [{idx}] {r}")

    backup = get_backup()
    if backup:
        backup.log_search(
            " ".join(p.get("title", "")[:30] for p in papers[:5]),
            list(AVAILABLE_SOURCES.keys()),
            result.get("total_found", 0),
            len(ingest_nums),
            result.get("duration_sec", 0),
        )


def run_upload(arg):
    uploader = get_uploader()
    if not uploader:
        print("请先登录用户。")
        return

    arg = arg.strip()
    if not arg:
        print("用法: upload <pdf_path> | upload --dir <dir> | upload inbox scan | upload inbox status")
        return

    if arg == "inbox scan":
        print("扫描收件箱...")
        results = uploader.scan_inbox()
        if not results:
            print("收件箱为空，无文件待处理。")
            return
        ok = sum(1 for r in results if r.get("status") == "ok")
        fail = sum(1 for r in results if r.get("status") != "ok" and r.get("status") != "skip")
        print(f"处理完成: {ok} 成功, {fail} 失败 (共 {len(results)} 个文件)")
        backup = get_backup()
        for r in results:
            icon = "[OK]" if r.get("status") == "ok" else "[FAIL]" if r.get("status") != "skip" else "[SKIP]"
            print(f"  {icon} {r.get('file_name', '?')}: {r.get('message', '')}")
            if backup and r.get("status") == "ok":
                backup.log_upload(
                    r.get("file_name", "?"),
                    r.get("title", r.get("message", "")),
                    r.get("paper_id", 0),
                    abstract=r.get("abstract", ""),
                    core_claim=r.get("core_claim", ""),
                )
        return

    if arg == "inbox status":
        status = uploader.inbox_status()
        print(f"收件箱状态: 待处理 {status['pending']} | 处理中 {status['processing']} | 已完成 {status['processed']} | 失败 {status['failed']}")
        if status.get('pending_files'):
            for f in status['pending_files']:
                print(f"  - {f}")
        return

    if arg.startswith("--dir "):
        dir_path = arg[6:].strip()
        results = uploader.upload_directory(dir_path)
        ok = sum(1 for r in results if r.get("status") == "ok")
        print(f"批量上传完成: {ok}/{len(results)} 成功")
        return

    result = uploader.upload_file(arg, confirm=True)
    print(f"  {result.get('message', '')}")
    if result.get("status") == "ok":
        backup = get_backup()
        if backup:
            backup.log_upload(
                result.get("file_name", arg),
                result.get("title", result.get("message", "")),
                result.get("paper_id", 0),
                abstract=result.get("abstract", ""),
                core_claim=result.get("core_claim", ""),
            )


def run_download(arg):
    arxiv_id = arg.strip()
    if not arxiv_id:
        print("用法: download <arxiv_id>")
        return
    print(f"\n[DOWNLOAD] 下载论文: {arxiv_id}")
    result = download_paper.invoke({"arxiv_id": arxiv_id})
    try:
        data = json.loads(result) if isinstance(result, str) else result
        print(f"  {data.get('message', result)}")
    except Exception:
        print(result)


def run_progress(topic):
    storage = get_storage()
    if not storage:
        print('请先登录用户。')
        return
    print(f'\n[STATS] 研究进展: {topic}\n')
    from research_assistant.tools.progress import get_progress_summary
    summary = get_progress_summary.invoke({'topic': topic})
    try:
        data = json.loads(summary) if isinstance(summary, str) else summary
        print(f'总记录: {data.get("total_entries", 0)} 条')
        timeline = data.get('timeline', [])
        if timeline:
            for t in timeline:
                print(f'  {t}')
        else:
            print(data.get('message', ''))
    except Exception:
        print(summary)


def run_record(topic):
    storage = get_storage()
    if not storage:
        print('请先登录用户。')
        return
    print(f'\n[NOTE] 记录「{topic}」的研究进展')
    print('进展类型: experiment(实验) / reading(阅读) / idea(想法) / result(结果) / other(其他)')
    entry_type = input('类型 (默认other): ').strip() or 'other'
    title = input('标题: ').strip()
    if not title:
        print('标题不能为空。')
        return
    content = input('详细描述: ').strip()
    results = input('实验结果/数据 (可选): ').strip() or None
    insights = input('获得的洞察 (可选): ').strip() or None
    next_actions = input('下一步计划 (可选): ').strip() or None

    from research_assistant.tools.progress import record_user_progress
    result = record_user_progress.invoke({
        'topic': topic, 'entry_type': entry_type, 'title': title,
        'content': content, 'results': results,
        'insights': insights, 'next_actions': next_actions,
    })
    print(f'\n{result}')

    try:
        data = json.loads(result) if isinstance(result, str) else result
        entry = data.get('entry', {})
        storage.add_progress({
            'topic': topic, 'entry_type': entry_type, 'title': title,
            'content': content, 'results': results,
            'insights': insights, 'next_actions': next_actions,
        })
        vs = _get_vs()()
        vs.upsert(user_manager.current_user.username, 'progress', [{
            'id': str(uuid.uuid5(uuid.NAMESPACE_DNS, f"progress_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")),
            'text': f"{title} {content} {results or ''} {insights or ''}",
            'payload': {'topic': topic, 'type': entry_type, 'title': title},
        }])
        suggestions = data.get('suggestions', {})
        if suggestions.get('suggested_searches'):
            print(f'\n  建议搜索: {" / ".join(suggestions["suggested_searches"])}')
        if suggestions.get('suggested_experiments'):
            print(f'  建议实验: {" / ".join(suggestions["suggested_experiments"])}')
    except Exception:
        logger.warning("后台入库异常", exc_info=True)
        print(f'  [WARN] 后台入库异常，详情见日志')

    backup = get_backup()
    if backup:
        backup.log_progress(
            topic, entry_type, title,
            content=content, results=results or "",
            insights=insights or "", next_actions=next_actions or "",
        )


def run_recall(arg):
    ep = get_episodic()
    if not ep:
        print('请先登录用户。')
        return
    query = arg.strip()
    if not query:
        print('用法: recall <你想起什么>')
        print('例如: recall 我们之前关于钝化讨论了什么')
        return
    print(f'\n[RECALL] 搜索记忆: {query}\n')

    mem_results = ep.recall_context(query, limit=10)
    from research_assistant.rag.vector_store import VectorStore
    vs = VectorStore()
    prog_results = vs.search(user_manager.current_user.username, 'progress', query, limit=10)

    merged = []
    for r in mem_results:
        merged.append(('memory', r.get('score', 0), r))
    for r in prog_results:
        merged.append(('progress', r.get('score', 0), r))
    merged.sort(key=lambda x: x[1], reverse=True)

    if not merged:
        print('未找到相关记忆。')
        return

    questions = ep.get_unresolved_questions(limit=5)
    for source, score, r in merged[:15]:
        payload = r.get('payload', {})
        text = payload.get('text', '')
        if text:
            tag = '[会话]' if source == 'memory' else '[进展]'
            print(f'  {tag} [匹配度: {score:.2f}] {text[:300]}')
            print('  ---')

    if questions:
        print(f'\n未解决的问题:')
        for q in questions:
            print(f'  - {q}')


def run_qa(arg):
    qa = get_qa()
    if not qa:
        print('请先登录用户。')
        return

    print()
    result = qa.ask(arg)

    print(f'\n{"=" * 60}')
    print(result['answer'])
    print(f'{"=" * 60}')

    cited = result.get('cited_papers', [])
    if cited:
        print(f'\n📚 引用论文: {len(cited)}篇')
        for i, p in enumerate(cited[:5], 1):
            print(f'  [{i}] {p.get("title","?")[:60]}')

    directions = result.get('directions', [])
    if directions:
        print(f'\n📋 后续科研方向:')
        for i, d in enumerate(directions, 1):
            print(f'  {i}. [{d.get("priority", "?")}] {d.get("title", "")}')
            print(f'     {d.get("description", "")}')

    if result.get('progress_recorded'):
        print(f'\n✅ 后续方向已自动记录到「研究进展」')
    elif directions:
        print(f'\n💡 输入 record <主题> 可手动记录这些方向')


def run_backup():
    backup = get_backup()
    if not backup:
        print('请先登录用户。')
        return
    result = backup.full_backup()
    print(f'\n[BACKUP] {result}')


def run_user_cmd(arg):
    parts = arg.split()
    if not parts:
        print('用法: user list | user switch <name> | user delete <name>')
        return
    sub = parts[0].lower()
    if sub == 'list':
        users = user_manager.list_users()
        if users:
            print(f'用户列表:')
            for u in users:
                marker = ' <-- 当前' if u == user_manager.current_user.username else ''
                print(f'  - {u}{marker}')
        else:
            print('暂无注册用户。')
    elif sub == 'switch' and len(parts) > 1:
        user_manager.switch_user(parts[1])
        print(f'已切换到用户: {parts[1]}')
    elif sub == 'delete' and len(parts) > 1:
        target = parts[1]
        if target == user_manager.current_user.username:
            confirm = input(f"你正在删除当前登录用户 '{target}'，所有数据将被永久删除！输入用户名确认: ").strip()
            if confirm != target:
                print("已取消。")
                return
        else:
            confirm = input(f"确认永久删除用户 '{target}' 及其全部数据？(y/N): ").strip().lower()
            if confirm != 'y':
                print("已取消。")
                return
        if user_manager.delete_user(target):
            print(f"用户 '{target}' 已删除。")
        else:
            print(f"用户 '{target}' 不存在。")
    else:
        print('用法: user list | user switch <name> | user delete <name>')


def run_paper_cmd(arg):
    """paper list | paper delete <id> | paper rebuild <id>"""
    parts = arg.split()
    storage = get_storage()
    if not storage:
        print('请先登录用户。')
        return

    if not parts:
        print('用法: paper list | paper delete <id> | paper rebuild <id>')
        return

    sub = parts[0].lower()

    if sub == 'list':
        papers = storage.get_all_papers()
        if not papers:
            print('知识库为空。')
            return
        print(f'\n知识库: {len(papers)} 篇论文\n')
        for p in papers:
            pid = p.get('id', '?')
            title = (p.get('title', '(无标题)') or '(无标题)')[:60]
            year = p.get('year', '?')
            src = p.get('source', '?')
            quality = p.get('annotation_quality', '?')
            print(f'  [{pid}] {year} | {src} | {quality}')
            print(f'       {title}')

    elif sub == 'delete' and len(parts) > 1:
        try:
            pid = int(parts[1])
        except ValueError:
            print(f'无效 ID: {parts[1]}')
            return
        p = storage.get_paper(pid)
        if not p:
            print(f'论文 ID={pid} 不存在。')
            return
        title = (p.get('title', '') or '(无标题)')[:60]
        confirm = input(f'确认删除 [{pid}] {title}? (y/N): ').strip().lower()
        if confirm != 'y':
            print('已取消。')
            return
        # 删 SQLite
        storage.delete_paper(pid)
        # 删 Qdrant（删除该 paper 的所有 chunk）
        try:
            from research_assistant.rag.vector_store import VectorStore
            vs = VectorStore()
            vs.delete_by_paper_id(user_manager.current_user.username, "papers", pid)
        except Exception:
            pass
        print(f'已删除: [{pid}] {title}')

    elif sub == 'rebuild' and len(parts) > 1:
        try:
            pid = int(parts[1])
        except ValueError:
            print(f'无效 ID: {parts[1]}')
            return
        p = storage.get_paper(pid)
        if not p:
            print(f'论文 ID={pid} 不存在。')
            return
        print(f'重建 [{pid}] {(p.get("title","") or "(无标题)")[:60]}')
        _rebuild_paper(pid, p, storage)
    elif sub == 'rebuild' and (len(parts) == 1 or parts[1] == 'all'):
        papers = storage.get_all_papers()
        print(f'重建全部 {len(papers)} 篇论文...')
        for i, p in enumerate(papers, 1):
            pid = p.get('id')
            print(f'  [{i}/{len(papers)}] ID={pid}')
            _rebuild_paper(pid, p, storage)
    else:
        print('用法: paper list | paper delete <id> | paper rebuild <id> | paper rebuild all')


def _rebuild_paper(paper_id: int, paper: dict, storage):
    """用当前分块策略重建单篇论文的向量。"""
    import os
    from research_assistant.rag.ingestion import IngestionPipeline
    from research_assistant.rag.vector_store import VectorStore

    # 重新加载全文
    file_path = paper.get("file_path", "")
    full_text = ""
    if file_path and os.path.exists(file_path):
        try:
            from research_assistant.loaders import load_document
            full_text, _ = load_document(file_path)
        except Exception:
            pass

    if not full_text:
        print(f'    无法获取全文，跳过。')
        return

    # 删旧向量
    vs = VectorStore()
    try:
        vs.delete_by_paper_id(user_manager.current_user.username, "papers", paper_id)
    except Exception:
        pass

    # 重新标注 + 分块 + 入库
    pipeline = IngestionPipeline(storage, user_manager.current_user.username)
    annotation = paper.get("annotation", {})
    if isinstance(annotation, str):
        import json
        try:
            annotation = json.loads(annotation)
        except Exception:
            annotation = {}
    if not annotation or not annotation.get("keywords_material"):
        annotation = pipeline.generate_annotation(paper, full_text)

    paper["annotation"] = annotation
    chunks = pipeline.chunk_paper(full_text, paper_id, annotation)
    if chunks:
        vs.upsert(user_manager.current_user.username, "papers", chunks)
        print(f'    重建: {len(chunks)} 个句子节点')
    else:
        chunks = pipeline.chunk_summary_only(paper, paper_id, annotation)
        vs.upsert(user_manager.current_user.username, "papers", chunks)
        print(f'    重建: 1 个摘要节点')


def run_review(topic):
    print_banner()
    storage = get_storage()
    if not storage:
        print("请先登录用户。")
        return
    username = user_manager.current_user.username
    checkpoint_db = str(BASE_DIR / "users" / username / "checkpoints.db")

    def kb_search(t, limit=10):
        retriever = _get_hr()(storage, username)
        return retriever.search_papers(t, limit=limit)

    def get_plan(t):
        return storage.get_active_plan(t)

    graph = _get_graph()(checkpoint_db)
    thread_id = f"review_{topic[:30]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    config = {"configurable": {"thread_id": thread_id}}
    set_runtime_context(thread_id,
        _kb_search_func=kb_search, _get_plan_func=get_plan,
        _storage=storage, _username=username,
    )
    initial_state = {"topic": topic, "additional_requirements": "", "messages": []}
    print(f"\n[START] 启动文献综述: {topic}\n")

    stage_labels = {
        "understand_intent": "理解意图", "search_papers": "搜索论文(三路合流)",
        "analyze_papers": "分层分析", "synthesize_review": "撰写综述",
        "plan_research": "制定计划", "user_progress_input": "等待进展输入",
        "assess_progress": "评估进展", "suggest_next": "下一步建议",
    }

    review_conclusion = ""
    try:
        step = 0
        for event in graph.stream(initial_state, config):
            step += 1
            node_name = list(event.keys())[0] if event else "unknown"
            node_data = event.get(node_name, {})
            stage = node_data.get("current_stage", node_name)
            print(f"\n{'='*60}")
            print(f"  步骤 {step}: {stage_labels.get(stage, stage)}")
            print(f"{'='*60}")

            if stage == "search_papers":
                papers = node_data.get("papers_found", [])
                kb_n = sum(1 for p in papers if p.get("source_label") == "kb")
                new_n = sum(1 for p in papers if p.get("source_label") == "new_search")
                print(f"  找到: {len(papers)} 篇 (KB:{kb_n} 新搜索:{new_n})")
            elif stage == "analyze_papers":
                analyzed = node_data.get("analyzed_papers", [])
                skipped = node_data.get("papers_skipped", [])
                deep_n = sum(1 for a in analyzed if a.get("analysis_level") == "deep")
                print(f"  深度:{deep_n} 轻量:{len(analyzed)-deep_n} 跳过:{len(skipped)}")
            elif stage == "synthesize_review":
                review_text = node_data.get("literature_review", "")
                if review_text:
                    # 取综述开头作为结论摘要（去除 markdown 标记）
                    import re
                    plain = re.sub(r"[#*>\-\n]+", " ", review_text[:600])
                    plain = re.sub(r"\s+", " ", plain).strip()
                    review_conclusion = plain[:300]
                print(f"  综述字数: {node_data.get('review_word_count', 0)}")
            elif stage == "user_progress_input":
                print("\n  [NOTE] 请输入研究进展 (skip跳过 quit退出)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    backup = get_backup()
    if backup:
        backup.log_review(topic, 0, 0, 0, 0, conclusion=review_conclusion)


def _on_quit():
    print("\n正在生成会话摘要...")
    qa = get_qa()
    if qa:
        # 紧急刷盘：防抖队列中未处理的事实立即提取
        try:
            qa.working.add_nowait_flush()
        except Exception:
            pass
        qa.end_session()
    ep = get_episodic()
    if ep:
        summary = ep.generate_session_summary()
        if summary and summary.get("topics"):
            print(f"\n会话要点:")
            for topic in summary.get("topics", []):
                print(f"  - {topic}")
            for disc in summary.get("key_discussions", []):
                print(f"  - {disc}")
            edit = input("\n要追加什么吗？(直接回车跳过): ").strip()
            if edit:
                summary["key_discussions"].append(edit)
            ep.save_session_summary(summary)
    backup = get_backup()
    if backup:
        backup.full_backup()
    print("\n再见！")
