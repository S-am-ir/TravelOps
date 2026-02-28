from contextlib import asynccontextmanager
from uuid import uuid4
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from src.graph import create_agent

# App State
_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = await create_agent()
    print("[API] Agent ready")
    yield
    print("[API] Shutting down")

app = FastAPI(
    title="TraveOps Agent API",
    description="Personal AI Travel admin for Nepal — travel, reminders, creative",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    all_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response models
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    thread_id: str
    response: str
    intent: Optional[str] = None
    interrupted: bool = False
    interrupt_data: Optional[dict] = None

class ConversationHistory(BaseModel):
    thread_id: str
    messages: list[dict]

# Helper
def _extract_response(state: dict) -> str:
    """Pull the last AI message from state, fallback to final response field"""
    final = state.get("final_response")
    if final:
        return final
    
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
        
    return "I've processed your request."

def _get_interrupt_data(state) -> Optional[dict]:
    """If graph is paused at an interrupt, extract the interrupt payload"""
    try:
        # LangGraph stores pending interrupt data in state.tasks
        tasks = getattr(state, "tasks", [])
        for task in tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value if task.interrupts else None
    except Exception:
        pass
    return None

# Routes
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the agent.

    New conversation: omit thread_id - a new one is assigned.
    Continue conversation: pass the same thread_id from the previous response.
    Confirm interrupt (e.g. WhatsApp send): pass the same thread_id; the API  
    auto-detects the paused state and resumes with your message.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    
    thread_id = request.thread_id or str(uuid4())
    config = {"configuration": {"thread_id": thread_id}}

    # Check if the graph is currently paused at an interrupt for this thread
    try:
        current_state = await _agent.aget_state(config)
        is_interrupted = bool(current_state.next) # has pending nodes = paused
    except Exception:
        is_interrupted = False

    try:
        if is_interrupted:
            # Resume from where it paused - pass user message as the resume value
            result = await _agent.ainvoke(
                Command(resume=request.message),
                config=config,
            )
        else:
            # New turn - append human message and run from entry point
            input_state: dict = {
                "messages": [HumanMessage(content=request.message),]
            }

            result = await _agent.ainvoke(input_state, config=config)
    
    except Exception as e:
        print(f"[API] Agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    
    # Check if we ended at a new interrupt (e.g. WhatsApp confirmation)
    try:
        post_state = await _agent.aget_state(config)
        still_interrupted = bool(post_state.next)
        interrupt_payload = _get_interrupt_data(post_state) if still_interrupted else None 
    except Exception:
        still_interrupted = False
        interrupt_payload = None
    
    response_text = _extract_response(result)
    # If paused at interrupt, surface the interrupt prompt to the user
    if still_interrupted and interrupt_payload and isinstance(interrupt_payload, dict):
        response_text = interrupt_payload.get("prompt", response_text)

    return ChatResponse(
        thread_id=thread_id,
        response=response_text,
        intent=result.get("intent"),
        interrupted=still_interrupted,
        interrupt_data=interrupt_payload,
    )

@app.get("/chat/{thread_id}", response_model=ConversationHistory)
async def get_history(thread_id: str):
    """Retrive the message history for a conversation thread."""
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
        serialized.append({
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content if hasattr(msg, "content") else str(msg),
        })

    return ConversationHistory(thread_id=thread_id, messages=serialized)

@app.delete("/chat/{thread_id}")
async def clear_thread(thread_id: str):
    """Clear a conversation thread (start fresh with the same ID)"""
    # Langgraph doesnt have a delete API; returning 200 so the client
    # can pass a new thread_id to effectively start fresh.
    return {"thread_id": thread_id, "cleared": True, "note": "Pass a new thread_id to start fresh"}

@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": _agent is not None}