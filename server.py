"""科研助手 REST API 服务 — FastAPI + SSE Streaming。

启动:
  python server.py
  或: uvicorn server:app --host 0.0.0.0 --port 8000 --reload

端点:
  POST /chat/stream     SSE 流式问答
  POST /chat            非流式问答
  POST /papers/search   外部论文搜索
  GET  /papers          论文列表
  DELETE /papers/{id}   删除论文
  GET  /health          健康检查
  GET  /docs            Swagger UI
"""

from __future__ import annotations
import sys, os
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="科研助手 API",
    description="多层 RAG Agent 系统 — 论文检索、智能问答、文献综述",
    version="1.0.0",
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from api.routes.chat import router as chat_router
from api.routes.search import router as search_router
from api.routes.papers import router as papers_router

app.include_router(chat_router)
app.include_router(search_router)
app.include_router(papers_router)


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok", "version": "1.0.0"}


# ── CLI 入口 ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
