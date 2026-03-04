import json
from datetime import datetime
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.types import interrupt
from src.agents.state import AgentState, ReminderExtraction
from src.mcp_internals.client import get_mcp_tools
from src.config.settings import settings

_scheduler = None

def _get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
        _scheduler.start()
        print("[Reminder] APScheduler started")
    return _scheduler


# ── LLM ───────────────────────────────────────────────────────────────────
# qwen3-32b: strong structured output, good at datetime parsing + context recall
# Falls back to llama-3.3-70b-versatile if qwen3 fails

def get_reminder_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        reasoning_effort="none",   # top-level param, NOT model_kwargs
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )

def get_reminder_fallback_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )


# ── System Prompt ─────────────────────────────────────────────────────────

REMINDER_SYSTEM = """You are extracting reminder details from a user's message.

Today (Nepal time): {today}

Extract:
- reminder_message: concise Telegram-ready text of what to remind about.
  If the user says "the trip plan you just made" or "send me the itinerary" or similar,
  look back through the conversation history and summarise the key details
  (destination, dates, flights, hotel, budget, activities).
- scheduled_for: ISO datetime (YYYY-MM-DDTHH:MM:SS) if a specific time was given,
  or "now" for immediate send. Nepal is UTC+5:45.
- repeat_rule: "daily", "weekly", or "none"

Respond with valid JSON only.
"""


# ── Tool result parser ────────────────────────────────────────────────────

def _parse_tool_result(raw) -> dict:
    """Normalize MCP tool result to a plain dict regardless of wrapper format."""
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, list):
        text = ""
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
            elif isinstance(block, str):
                text += block
        raw = text.strip()

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"status": "sent", "raw": raw}

    return {"status": "error", "error": f"Unexpected result type: {type(raw)}"}


# ── Node ──────────────────────────────────────────────────────────────────

async def reminder_agent_node(state: AgentState) -> dict:
    messages  = state.get("messages", [])
    now_iso   = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_msg = SystemMessage(content=REMINDER_SYSTEM.format(today=now_iso))

    extracted: ReminderExtraction | None = None
    for name, llm_factory in [("qwen3-32b", get_reminder_llm), ("llama-3.3-70b (fallback)", get_reminder_fallback_llm)]:
        try:
            llm = llm_factory()
            structured_llm = llm.with_structured_output(ReminderExtraction)
            extracted = await structured_llm.ainvoke([system_msg] + messages)
            print(f"[Reminder] Extracted via {name}: scheduled_for={extracted.scheduled_for}")
            break
        except Exception as e:
            print(f"[Reminder] {name} failed: {e}, trying next...")
            continue

    if extracted is None:
        err = "Couldn't parse your reminder. Please try again with a clearer time."
        return {"messages": [AIMessage(content=err)], "final_response": err}

    reminder_text = extracted.reminder_message
    scheduled_for = extracted.scheduled_for
    send_now      = scheduled_for.lower() == "now"

    run_dt = None
    if not send_now:
        try:
            run_dt = datetime.fromisoformat(scheduled_for)
        except ValueError:
            send_now = True

    # ── Human-in-the-loop confirmation ───────────────────────────────────
    # Pause and show the confirmation card in the UI before sending anything.
    # The frontend reads interrupt_data.draft and interrupt_data.to to render the card.
    confirmation = interrupt({
        "type":  "telegram_confirmation",
        "draft": reminder_text,
        "to":    "your Telegram",
        "prompt": f"Send this to your Telegram?\n\n{reminder_text}\n\nReply yes/no.",
    })

    confirmed = (
        isinstance(confirmation, dict) and confirmation.get("confirmed")
    ) or (
        isinstance(confirmation, str)
        and confirmation.strip().lower() in ("yes", "y", "confirm", "send")
    )

    if not confirmed:
        cancelled = "Cancelled — nothing was sent."
        return {"messages": [AIMessage(content=cancelled)], "final_response": cancelled}

    if send_now:
        response_text = await _send_telegram_now(reminder_text)
    else:
        response_text = await _schedule_reminder(
            message=reminder_text,
            run_dt=run_dt,
            repeat_rule=extracted.repeat_rule or "none",
        )

    return {
        "messages": [AIMessage(content=response_text)],
        "final_response": response_text,
    }


# ── Telegram helpers ──────────────────────────────────────────────────────

async def _send_telegram_now(message: str) -> str:
    tools   = await get_mcp_tools(servers=["comms"])
    tg_tool = next((t for t in tools if t.name == "send_telegram_message"), None)

    if not tg_tool:
        return "Telegram tool not available. Is the comms MCP server running?"

    try:
        raw         = await tg_tool.ainvoke({"body": message})
        print(f"[Reminder] Raw tool result type={type(raw)}: {repr(raw)[:200]}")
        result_dict = _parse_tool_result(raw)

        if result_dict.get("status") == "sent":
            return f"✅ Reminder sent to your Telegram!\n\nMessage: {message}"
        else:
            return f"Failed to send reminder: {result_dict.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Telegram send failed: {e}"


async def _schedule_reminder(message: str, run_dt: datetime, repeat_rule: str) -> str:
    scheduler = _get_scheduler()

    job_kwargs = dict(
        func=_fire_reminder,
        args=[message],
        id=f"reminder_{run_dt.timestamp():.0f}",
        replace_existing=True,
    )

    if repeat_rule == "daily":
        from apscheduler.triggers.cron import CronTrigger
        job_kwargs["trigger"] = CronTrigger(hour=run_dt.hour, minute=run_dt.minute)
        repeat_label = "daily"
    elif repeat_rule == "weekly":
        from apscheduler.triggers.cron import CronTrigger
        job_kwargs["trigger"] = CronTrigger(
            day_of_week=run_dt.strftime("%a").lower(),
            hour=run_dt.hour,
            minute=run_dt.minute,
        )
        repeat_label = f"every {run_dt.strftime('%A')}"
    else:
        from apscheduler.triggers.date import DateTrigger
        job_kwargs["trigger"] = DateTrigger(run_date=run_dt)
        repeat_label = "once"

    scheduler.add_job(**job_kwargs)
    friendly_time = run_dt.strftime("%A, %d %b %Y at %I:%M %p")
    return (
        f"🔔 Reminder scheduled ({repeat_label})!\n\n"
        f"Time: {friendly_time}\n"
        f"Message: {message}"
    )


async def _fire_reminder(message: str):
    print("[Reminder] Firing scheduled reminder")
    await _send_telegram_now(message)