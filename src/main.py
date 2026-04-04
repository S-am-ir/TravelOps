import asyncio
import os
from contextlib import asynccontextmanager
from uuid import uuid4
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from src.graph import create_agent
from src.config.settings import settings
from src.auth.service import (
    AuthService,
    init_auth_db,
    close_auth_db,
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserProfile,
)
from src.auth.middleware import get_current_user, get_optional_user

# App State
_agent = None
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _checkpointer

    # Ensure checkpoints directory exists (for SQLite reminders DB)
    import os

    os.makedirs("checkpoints", exist_ok=True)

    _agent, _checkpointer = await create_agent()
    try:
        await init_auth_db()
    except Exception as e:
        print(f"[API] Auth DB init failed (non-fatal): {e}")
    print("[API] Agent ready")
    yield
    try:
        await close_auth_db()
    except Exception:
        pass
    print("[API] Shutting down")


app = FastAPI(
    title="Travel Planner API",
    description="AI-powered travel planning, reminders, and creative tools",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow local dev + production frontend URL from env
cors_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    cors_origins.append(frontend_url)
# Allow all vercel.app subdomains for preview deployments
cors_origins.append("https://*.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ───────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    timezone: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    intent: Optional[str] = None
    interrupted: bool = False
    interrupt_data: Optional[dict] = None


class ConversationHistory(BaseModel):
    thread_id: str
    messages: list[dict]


# ── Helpers ─────────────────────────────────────────────────────────────


def _normalize_content(content) -> str:
    """Gemini 2.5-flash returns content as a list of blocks, not a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _extract_response(state: dict) -> str:
    """Pull the last AI message from state, fallback to final response field"""
    final = state.get("final_response")
    if final:
        return _normalize_content(final)

    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return _normalize_content(msg.content)

    return "I've processed your request."


def _get_interrupt_data(state) -> Optional[dict]:
    """If graph is paused at an interrupt, extract the interrupt payload"""
    try:
        tasks = getattr(state, "tasks", [])
        for task in tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value if task.interrupts else None
    except Exception:
        pass
    return None


# ── Auth routes ─────────────────────────────────────────────────────────


@app.post("/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    try:
        return await AuthService.register(req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    try:
        return await AuthService.login(req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/auth/me", response_model=UserProfile)
async def me(user: UserProfile = Depends(get_current_user)):
    return user


# ── Email SMTP config routes ──────────────────────────────────────────────


class EmailConfigRequest(BaseModel):
    app_password: str


@app.get("/settings/email")
async def get_email_settings(user: UserProfile = Depends(get_current_user)):
    smtp = await AuthService.get_smtp_config(user.id)
    if not smtp:
        return {"configured": False}
    return {
        "configured": True,
        "email": smtp["smtp_email"],
    }


@app.post("/settings/email")
async def save_email_settings(
    req: EmailConfigRequest,
    user: UserProfile = Depends(get_current_user),
):
    """Save SMTP credentials. Validates by attempting login to smtp.gmail.com."""
    import smtplib

    # Use the account email as the sending email
    send_email = user.email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(send_email, req.app_password)
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(
            status_code=400,
            detail="Invalid app password. Make sure you generated an App Password, not your regular password.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection failed: {e}")

    await AuthService.save_smtp_config(user.id, send_email, req.app_password)
    return {"configured": True, "email": send_email}


@app.delete("/settings/email")
async def delete_email_settings(user: UserProfile = Depends(get_current_user)):
    await AuthService.clear_smtp_config(user.id)
    return {"configured": False, "message": "Email config removed"}


# ── Chat routes ─────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: Optional[UserProfile] = Depends(get_optional_user),
):
    """
    Send a message to the agent.
    Authenticated users get persistent threads (auto-restored on login).
    Unauthenticated users get ephemeral threads (lost on session end).
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    # Determine thread_id
    if request.thread_id:
        thread_id = request.thread_id
    elif user and user.active_thread_id:
        thread_id = user.active_thread_id
    else:
        thread_id = str(uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    # Check if the graph is currently paused at an interrupt for this thread
    try:
        current_state = await _agent.aget_state(config)
        is_interrupted = bool(current_state.next)
    except Exception:
        is_interrupted = False

    try:
        if is_interrupted:
            import json as _json

            try:
                resume_value = _json.loads(request.message)
            except Exception:
                resume_value = request.message
            result = await _agent.ainvoke(
                Command(resume=resume_value),
                config=config,
            )
        else:
            from datetime import datetime
            local_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            if request.timezone:
                try:
                    import zoneinfo
                    tz = zoneinfo.ZoneInfo(request.timezone)
                    local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            input_state: dict = {
                "messages": [
                    HumanMessage(content=request.message),
                ],
                "user_id": user.id if user else None,
                "user_timezone": request.timezone,
                "user_local_time": local_time,
            }
            result = await _agent.ainvoke(input_state, config=config)

    except BaseException as e:
        import traceback
        error_details = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"[API] Agent error: {error_details}")
        
        err_msg = str(e)
        if hasattr(e, "exceptions") and e.exceptions:
            err_msg = f"{e}: {e.exceptions[0]}"
        raise HTTPException(status_code=500, detail=f"Agent error: {err_msg}")

    # Check if we ended at a new interrupt
    try:
        post_state = await _agent.aget_state(config)
        still_interrupted = bool(post_state.next)
        interrupt_payload = (
            _get_interrupt_data(post_state) if still_interrupted else None
        )
    except Exception:
        still_interrupted = False
        interrupt_payload = None

    response_text = _extract_response(result)
    if still_interrupted and interrupt_payload and isinstance(interrupt_payload, dict):
        response_text = interrupt_payload.get("prompt", response_text)

    # Persist thread_id for authenticated users
    if user:
        try:
            await AuthService.update_active_thread(user.id, thread_id)
        except Exception as e:
            print(f"[API] Failed to persist thread: {e}")

    return ChatResponse(
        thread_id=thread_id,
        response=response_text,
        intent=result.get("intent"),
        interrupted=still_interrupted,
        interrupt_data=interrupt_payload,
    )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: Optional[UserProfile] = Depends(get_optional_user),
):
    """Stream chat responses token-by-token via SSE."""
    from fastapi.responses import StreamingResponse
    from src.agents.nodes.Travel import token_callback
    import json as _json

    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    # Determine thread_id
    if request.thread_id:
        thread_id = request.thread_id
    elif user and user.active_thread_id:
        thread_id = user.active_thread_id
    else:
        thread_id = str(uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    # Check for existing interrupt
    try:
        current_state = await _agent.aget_state(config)
        is_interrupted = bool(current_state.next)
    except Exception:
        is_interrupted = False

    async def event_generator():
        token_queue = asyncio.Queue()

        async def put_token(t: str):
            await token_queue.put(t)

        # Set the token callback via contextvar
        token_callback.set(put_token)

        async def run_agent():
            try:
                if is_interrupted:
                    import json as __json

                    try:
                        resume_value = __json.loads(request.message)
                    except Exception:
                        resume_value = request.message
                    result = await _agent.ainvoke(
                        Command(resume=resume_value), config=config
                    )
                else:
                    from datetime import datetime
                    local_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    if request.timezone:
                        try:
                            import zoneinfo
                            tz = zoneinfo.ZoneInfo(request.timezone)
                            local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            pass

                    input_state = {
                        "messages": [HumanMessage(content=request.message)],
                        "user_id": user.id if user else None,
                        "user_timezone": request.timezone,
                        "user_local_time": local_time,
                    }
                    result = await _agent.ainvoke(input_state, config=config)
                await token_queue.put(("__DONE__", result))
            except BaseException as e:
                import traceback
                error_details = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                print(f"[API] Error in run_agent: {error_details}")
                # Try to extract the first sub-exception if it's an ExceptionGroup
                err_msg = str(e)
                if hasattr(e, "exceptions") and e.exceptions:
                    err_msg = f"{e}: {e.exceptions[0]}"
                await token_queue.put(("__ERROR__", err_msg))
            finally:
                token_callback.set(None)

        agent_task = asyncio.create_task(run_agent())

        # Stream tokens as they arrive
        while True:
            try:
                item = await asyncio.wait_for(token_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if agent_task.done():
                    # Drain remaining tokens
                    while not token_queue.empty():
                        item = await token_queue.get()
                        if isinstance(item, tuple):
                            break
                        yield f"data: {_json.dumps({'type': 'token', 'content': item})}\n\n"
                    break
                continue

            if isinstance(item, tuple) and item[0] == "__DONE__":
                _, result = item
                # Check for interrupt
                try:
                    post_state = await _agent.aget_state(config)
                    still_interrupted = bool(post_state.next)
                    interrupt_payload = None
                    if still_interrupted:
                        tasks = getattr(post_state, "tasks", [])
                        for task in tasks:
                            if hasattr(task, "interrupts") and task.interrupts:
                                interrupt_payload = task.interrupts[0].value
                                break
                except Exception:
                    still_interrupted = False
                    interrupt_payload = None

                final_response = _extract_response(result)
                if (
                    still_interrupted
                    and interrupt_payload
                    and isinstance(interrupt_payload, dict)
                ):
                    final_response = interrupt_payload.get("prompt", final_response)

                # Persist thread
                if user:
                    try:
                        await AuthService.update_active_thread(user.id, thread_id)
                    except Exception:
                        pass

                yield f"data: {_json.dumps({'type': 'done', 'thread_id': thread_id, 'response': final_response, 'intent': result.get('intent'), 'interrupted': still_interrupted, 'interrupt_data': interrupt_payload})}\n\n"
                break

            elif isinstance(item, tuple) and item[0] == "__ERROR__":
                yield f"data: {_json.dumps({'type': 'error', 'message': item[1]})}\n\n"
                break
            else:
                yield f"data: {_json.dumps({'type': 'token', 'content': item})}\n\n"

        await agent_task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chat/{thread_id}", response_model=ConversationHistory)
async def get_history(thread_id: str):
    """Retrieve the message history for a conversation thread."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await _agent.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thread not found: {e}")

    messages = state.values.get("messages", [])
    serialized = []
    for msg in messages:
        serialized.append(
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content if hasattr(msg, "content") else str(msg),
            }
        )

    return ConversationHistory(thread_id=thread_id, messages=serialized)


@app.delete("/chat/{thread_id}")
async def clear_thread(
    thread_id: str,
    user: Optional[UserProfile] = Depends(get_optional_user),
):
    """Clear a conversation thread and all its checkpoint data."""
    if _checkpointer is None:
        raise HTTPException(status_code=503, detail="Checkpointer not ready")

    # Clear active thread for authenticated user
    if user:
        try:
            current_thread = await AuthService.get_active_thread(user.id)
            if current_thread == thread_id:
                await AuthService.clear_active_thread(user.id)
        except Exception as e:
            print(f"[API] Failed to clear user thread: {e}")

    # Clear checkpointer state
    cleared = False

    # Try native delete method
    if hasattr(_checkpointer, "adelete_thread"):
        try:
            await _checkpointer.adelete_thread(thread_id)
            cleared = True
        except Exception as e:
            print(f"[API] adelete_thread failed: {e}")

    # Also try direct SQL if adelete_thread didn't work or as extra cleanup
    if not cleared and hasattr(_checkpointer, "conn"):
        try:
            pool = _checkpointer.conn
            async with pool.connection() as conn:
                for table in ["checkpoint_writes", "checkpoint_blobs", "checkpoints"]:
                    await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)
                    )
            cleared = True
        except Exception as e:
            print(f"[API] SQL cleanup failed: {e}")

    # For MemorySaver
    if not cleared and hasattr(_checkpointer, "_storage"):
        keys_to_delete = (
            [k for k in _checkpointer._storage if k[0] == thread_id]
            if isinstance(next(iter(_checkpointer._storage), None), tuple)
            else [k for k in _checkpointer._storage if k == thread_id]
        )
        for k in keys_to_delete:
            del _checkpointer._storage[k]
        cleared = True

    return {"thread_id": thread_id, "cleared": cleared}


# ── Health ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return "OK"


# ── Frontend serving ────────────────────────────────────────────────────

app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")


@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn

    print("[DEBUG] Starting FastAPI server...")
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print("[CRITICAL] Failed to start server:", str(e))
        raise
