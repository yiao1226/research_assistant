"""POST /papers/search — 外部论文搜索。"""

from __future__ import annotations
import asyncio
from fastapi import APIRouter, HTTPException

from api.schemas.search import SearchRequest
from api.deps import get_storage

router = APIRouter(tags=["search"])


@router.post("/papers/search")
async def search_papers(req: SearchRequest):
    """搜索外部论文（ArXiv + Semantic Scholar）。"""
    try:
        storage = get_storage(req.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"用户初始化失败: {e}")

    from research_assistant.tools.search_orchestrator import SearchOrchestrator
    orch = SearchOrchestrator(storage, req.username)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: orch.search(req.query, sources=req.sources, sort_by=req.sort_by),
    )

    return {
        "papers": result.get("papers", []),
        "total_found": result.get("total_found", 0),
        "search_focus": result.get("search_focus", ""),
        "duration_sec": result.get("duration_sec", 0),
    }
