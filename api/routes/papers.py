"""论文管理端点。"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query

from api.deps import get_storage

router = APIRouter(tags=["papers"])


@router.get("/papers")
async def list_papers(
    username: str = Query(default="eval"),
    limit: int = Query(default=50, le=200),
):
    """获取知识库论文列表。"""
    storage = get_storage(username)
    papers = storage.get_all_papers()[:limit]
    return {
        "total": len(papers),
        "papers": [
            {
                "id": p.get("id"),
                "title": p.get("title", "")[:100],
                "year": p.get("year"),
                "source": p.get("source"),
                "annotation_quality": p.get("annotation_quality", "none"),
            }
            for p in papers
        ],
    }


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: int, username: str = Query(default="eval")):
    """删除一篇论文（SQLite + Qdrant）。"""
    storage = get_storage(username)
    paper = storage.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"论文 ID={paper_id} 不存在")

    storage.delete_paper(paper_id)
    # 删除 Qdrant 中的向量
    try:
        from research_assistant.rag.vector_store import VectorStore
        vs = VectorStore()
        vs.delete_by_paper_id(username, "papers", paper_id)
    except Exception:
        pass

    return {"status": "ok", "message": f"已删除: {paper.get('title', '?')[:60]}"}
