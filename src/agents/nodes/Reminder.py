import json
from datetime import datetime
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.types import interrupt
from src.agents.state import AgentState, ReminderExtraction
from src.mcp_internals.client import get_mcp_tools
from src.model_api import invoke_with_fallback


_scheduler = None

def _get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
        _scheduler.start()
        print("[Reminder] APScheduler started")
    return _scheduler


REMINDER_SYSTEM = """You are extracting reminder details from a user's message.

Today (Nepal time): {today}

Extract:
- reminder_message: concise Telegram-ready text of what to remind about.
  If the user says "the trip plan you just made" or "send me the itinerary",
  look back through the conversation history and summarise the key details
  (destination, dates, flights, hotel, budget, activities).
- scheduled_for: ISO datetime (YYYY-MM-DDTHH:MM:SS) if a specific time was given,
  or "now" for immediate send. Nepal is UTC+5:45.
- repeat_rule: "daily", "weekly", or "none"

Respond with valid JSON only.
"""


def _parse_tool_result(raw) -> dict:
    """Normalise MCP tool result to a plain dict regardless of wrapper format."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        text = "".join(
            b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else (b if isinstance(b, str) else "")
            for b in raw
        ).strip()
        raw = text
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"status": "sent", "raw": raw}
    return {"status": "error", "error": f"Unexpected result type: {type(raw)}"}


async def reminder_agent_node(state: AgentState) -> dict:
    messages  = state.get("messages", [])
    now_iso   = datetime.now().strftime("%Y-%m-%d %H:%M")
    system_msg = SystemMessage(content=REMINDER_SYSTEM.format(today=now_iso))

    try:
        extracted, model = await invoke_with_fallback(
            [system_msg] + messages,
            structured_schema=ReminderExtraction,
        )
        print(f"[Reminder] Extracted via {model}: scheduled_for={extracted.scheduled_for}")
    except Exception as e:
        err = f"Couldn't parse your reminder: {e}. Please try again with a clearer time."
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

    # Human-in-the-loop: pause and show confirmation card before sending.
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


async def _send_telegram_now(message: str) -> str:
    tools   = await get_mcp_tools(servers=["comms"])
    tg_tool = next((t for t in tools if t.name == "send_telegram_message"), None)

    if not tg_tool:
        return "Telegram tool not available. Is the comms MCP server running?"

    try:
        raw         = await tg_tool.ainvoke({"body": message})
        result_dict = _parse_tool_result(raw)
        if result_dict.get("status") == "sent":
            return f"✅ Reminder sent to your Telegram!\n\nMessage: {message}"
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