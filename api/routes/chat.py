"""POST /chat/stream — SSE 流式问答端点。

面试要点:
  - SSE 事件不是裸字符串，而是类型化 JSON（thinking/tool_call/token/.../done）
  - 覆盖 CLI Hook，将 print → SSE event
  - async generator + asyncio.to_thread 实现同步 QAService 的异步化
  - X-Accel-Buffering: no 防止 Nginx 代理缓冲 SSE
"""

from __future__ import annotations
import asyncio, json, sys, io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.chat import ChatRequest
from api.deps import get_qa_service

router = APIRouter(tags=["chat"])


def _sse(event: str, data: dict) -> str:
    """格式化一条 SSE 消息。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式问答。

    事件类型:
      thinking   — Agent 开始分析
      tool_call  — 工具被调用
      tool_result— 工具返回结果
      token      — 逐 token（由 Hook 注入）
      citation   — 引用论文
      done       — 完成（含统计）
      error      — 异常
    """
    try:
        qa = get_qa_service(req.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化 QA 失败: {e}")

    # 注册 SSE Hook（覆盖 CLI 的 print-based hook）
    sse_queue: asyncio.Queue = asyncio.Queue()
    # 捕获主线程的 event loop（线程池线程里没有 loop）
    main_loop = asyncio.get_running_loop()

    def sse_thinking(question: str, *, _loop=main_loop):
        """agent_start hook → SSE thinking"""
        _loop.call_soon_threadsafe(
            lambda: sse_queue.put_nowait(
                _sse("thinking", {"message": f"分析: {question[:60]}..."})
            )
        )

    def sse_tool_call(name: str, args: dict, *, _loop=main_loop):
        """pre_tool hook → SSE tool_call"""
        preview = {k: str(v)[:60] for k, v in args.items()}
        _loop.call_soon_threadsafe(
            lambda: sse_queue.put_nowait(
                _sse("tool_call", {"tool": name, "args": preview})
            )
        )
        return None  # 不拦截

    def sse_tool_result(_name: str, _args: dict, result: str, *, _loop=main_loop):
        """post_tool hook → SSE tool_result"""
        summary = str(result)[:200].replace("\n", " ")
        _loop.call_soon_threadsafe(
            lambda: sse_queue.put_nowait(
                _sse("tool_result", {"tool": _name, "summary": summary})
            )
        )

    # 注入 SSE hooks + on_token 回调（注册后在子线程中实时推事件到队列）
    from research_assistant.agent.qa import register_hook
    register_hook("agent_start", sse_thinking)
    register_hook("pre_tool", sse_tool_call)
    register_hook("post_tool", sse_tool_result)

    # on_token: LLM 每生成一个 token → 推到 SSE + 打印到服务端终端
    def _sse_on_token(text: str):
        sys.stdout.write(text)
        sys.stdout.flush()
        main_loop.call_soon_threadsafe(
            lambda t=text: sse_queue.put_nowait(_sse("token", {"delta": t}))
        )

    async def event_generator():
        # 并发运行：QA 在后台执行 + 实时 drain SSE 事件队列
        import concurrent.futures
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        try:
            # 提交 QA 任务（on_token 实时推 token 事件到队列）
            future = main_loop.run_in_executor(
                pool, qa.ask, req.message, _sse_on_token,
            )

            # 实时 drain queue，直到 QA 完成
            while not future.done():
                try:
                    event_text = await asyncio.wait_for(sse_queue.get(), timeout=0.1)
                    yield event_text
                except asyncio.TimeoutError:
                    continue

            # QA 完成，获取结果
            result = future.result()

            # drain 残留事件（包括最后几个 token）
            while not sse_queue.empty():
                yield sse_queue.get_nowait()

            # 发送引用论文
            cited = result.get("cited_papers", [])
            if cited:
                yield _sse("citation", {"papers": [
                    {"title": p.get("title", "?")[:80]} for p in cited[:5]
                ]})

            # 完成（token 已通过 on_token 实时发出，不再重复）
            tool_calls = len(result.get("tool_calls", []))
            yield _sse("done", {
                "tool_calls": tool_calls,
                "cited_papers": len(cited),
            })

        except Exception as e:
            yield _sse("error", {"code": "QA_FAILED", "message": str(e)})
        finally:
            pool.shutdown(wait=False)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 代理缓冲
        },
    )


@router.post("/chat")
async def chat_sync(req: ChatRequest):
    """非流式问答（一次性返回）。"""
    try:
        qa = get_qa_service(req.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化 QA 失败: {e}")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, qa.ask, req.message)
    answer = result.get("answer", "")
    cited = result.get("cited_papers", [])
    tool_calls = result.get("tool_calls", [])

    return {
        "answer": answer,
        "cited_papers": [{"title": p.get("title", "?")[:80]} for p in cited[:5]],
        "tool_calls": [{"tool": tc["tool"], "args": tc.get("args", {})} for tc in tool_calls],
    }
