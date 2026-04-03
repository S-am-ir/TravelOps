import json
import asyncio
import threading
from datetime import datetime
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.types import interrupt
from src.agents.state import AgentState, ReminderExtraction
from src.model_api import invoke_with_fallback


# ── Dedicated event loop thread for async job execution ─────────────────
# APScheduler 3.x does NOT await async job functions — it calls them
# synchronously and discards the returned coroutine.  We work around this
# by running a permanent background event loop in a daemon thread and
# submitting async coroutines to it from sync job callbacks.

_loop = asyncio.new_event_loop()


def _start_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


_thread = threading.Thread(target=_start_loop, daemon=True, name="reminder-loop")
_thread.start()


# ── Scheduler singleton with persistent job store ───────────────────────
_scheduler: Optional[AsyncIOScheduler] = None


def _get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        try:
            jobstores = {
                "default": SQLAlchemyJobStore(
                    url="sqlite:///checkpoints/reminders.db",
                    tablename="apscheduler_jobs",
                )
            }
        except Exception as e:
            print(
                f"[Reminder] SQLite job store failed ({e}), falling back to in-memory"
            )
            jobstores = {}

        _scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 3600,
            },
        )
        _scheduler.start()
        print("[Reminder] APScheduler started (persistent job store)")
    return _scheduler


# ── Reminder system prompt ──────────────────────────────────────────────

REMINDER_SYSTEM = """You are extracting reminder details from a user's message.

Today (Local time): {today}
User Timezone: {timezone}

Extract:
- reminder_message: concise text of what to remind about.
  If the user says "the trip plan you just made" or "send me the itinerary",
  look back through the conversation history and summarise the key details
  (destination, dates, flights, hotel, budget, activities).
- scheduled_for: ISO datetime (YYYY-MM-DDTHH:MM:SS) if a specific time was given,
  or "now" for immediate send. Use the user's local timezone.
- recipient_email: email address(es) to send to. If user mentions MULTIPLE emails,
  include ALL of them as comma-separated (e.g. "user1@gmail.com, user2@gmail.com").
  Null if not mentioned — it will go to the user's own email.
- repeat_rule: "daily", "weekly", or "none"

Respond with valid JSON only.
"""


# ── Helpers ─────────────────────────────────────────────────────────────


def _parse_tool_result(raw) -> dict:
    """Normalise MCP tool result to a plain dict regardless of wrapper format."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        text = "".join(
            b.get("text", "")
            if isinstance(b, dict) and b.get("type") == "text"
            else (b if isinstance(b, str) else "")
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


# ── Main agent node ────────────────────────────────────────────────────


async def reminder_agent_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    now_iso = state.get("user_local_time") or datetime.now().strftime("%Y-%m-%d %H:%M")
    tz_info = state.get("user_timezone") or "UTC"
    system_msg = SystemMessage(content=REMINDER_SYSTEM.format(today=now_iso, timezone=tz_info))

    try:
        extracted, model = await invoke_with_fallback(
            [system_msg] + messages,
            structured_schema=ReminderExtraction,
        )
        print(
            f"[Reminder] Extracted via {model}: scheduled_for={extracted.scheduled_for}"
        )
    except Exception as e:
        err = (
            f"Couldn't parse your reminder: {e}. Please try again with a clearer time."
        )
        return {"messages": [AIMessage(content=err)], "final_response": err}

    reminder_text = extracted.reminder_message
    scheduled_for = extracted.scheduled_for
    send_now = scheduled_for.lower() == "now"

    run_dt = None
    if not send_now:
        try:
            run_dt = datetime.fromisoformat(scheduled_for)
        except ValueError:
            send_now = True

    # ── Human-in-the-loop: pause HERE (at scheduling time) ──────────────
    # Determine recipient email
    user_id = state.get("user_id")
    recipient = None
    if user_id:
        try:
            from src.auth.service import AuthService

            smtp_config = await AuthService.get_smtp_config(user_id)
            if smtp_config:
                recipient = smtp_config["smtp_email"]
        except Exception:
            pass

    # Override with explicitly specified email(s)
    if extracted.recipient_email:
        recipient = extracted.recipient_email

    to_label = recipient if recipient else "your email"

    # Parse for display
    to_list = (
        [r.strip() for r in recipient.replace(";", ",").split(",")]
        if recipient
        else ["your email"]
    )
    to_display = ", ".join(to_list)

    confirmation = interrupt(
        {
            "type": "email_confirmation",
            "draft": reminder_text,
            "to": to_display,
            "prompt": f"Send this email reminder to {to_display}?\n\n{reminder_text}\n\nReply yes/no.",
        }
    )

    confirmed = (isinstance(confirmation, dict) and confirmation.get("confirmed")) or (
        isinstance(confirmation, str)
        and confirmation.strip().lower() in ("yes", "y", "confirm", "send")
    )

    if not confirmed:
        cancelled = "Cancelled — nothing was sent."
        return {"messages": [AIMessage(content=cancelled)], "final_response": cancelled}

    if send_now:
        response_text = await _send_email_now(user_id, reminder_text, recipient)
    else:
        response_text = await _schedule_reminder(
            user_id=user_id,
            message=reminder_text,
            run_dt=run_dt,
            repeat_rule=extracted.repeat_rule or "none",
            recipient=recipient,
        )

    return {
        "messages": [AIMessage(content=response_text)],
        "final_response": response_text,
    }


# ── Email send (immediate) ─────────────────────────────────────────────


async def _send_email_now(user_id: str, message: str, recipient: str = None) -> str:
    """Send an email reminder via SMTP. Supports comma-separated multiple recipients."""
    from src.email_service import send_email_gmail
    from src.auth.service import AuthService

    print(f"[Reminder] _send_email_now called: recipient={recipient!r}")

    if not user_id:
        return "No user session — please log in and set up email in Settings to send reminders."

    smtp_config = await AuthService.get_smtp_config(user_id)
    if not smtp_config:
        return (
            "Email isn't set up yet.\n\n"
            "To send email reminders:\n"
            "1. Open **Settings** (gear icon in the header)\n"
            "2. Enter your email and Gmail App Password\n"
            "3. Come back and ask me again!"
        )

    from_email = smtp_config["smtp_email"]

    # Parse recipients — handle comma-separated emails
    if recipient:
        recipients = [
            r.strip() for r in recipient.replace(";", ",").split(",") if r.strip()
        ]
    else:
        recipients = [from_email]

    results = []
    for to_email in recipients:
        result = send_email_gmail(
            gmail_email=from_email,
            app_password=smtp_config["smtp_password"],
            to_email=to_email,
            subject="Reminder",
            body=message,
        )
        if result["status"] == "sent":
            results.append(f"Sent to {to_email}")
        else:
            results.append(f"Failed for {to_email}: {result.get('error', 'Unknown')}")

    return f"Reminder:\n\n{message}\n\n" + "\n".join(results)


# ── APScheduler callbacks ──────────────────────────────────────────────


def _fire_reminder(user_id: str, message: str, recipient: str = None):
    """Sync wrapper called by APScheduler. Delegates to the background loop."""
    print(f"[Reminder] Firing scheduled reminder: {message[:80]}...")
    asyncio.run_coroutine_threadsafe(
        _send_email_now(user_id, message, recipient), _loop
    )


async def _schedule_reminder(
    user_id: str,
    message: str,
    run_dt: datetime,
    repeat_rule: str,
    recipient: str = None,
) -> str:
    # Check email config BEFORE scheduling
    from src.auth.service import AuthService

    smtp_config = await AuthService.get_smtp_config(user_id)
    if not smtp_config:
        return (
            "Email isn't set up yet, so I can't schedule reminders.\n\n"
            "To set up email reminders:\n"
            "1. Open **Settings** (gear icon in the header)\n"
            "2. Enter your email and Gmail App Password\n"
            "3. Come back and ask me again!"
        )

    scheduler = _get_scheduler()

    if repeat_rule == "daily":
        from apscheduler.triggers.cron import CronTrigger

        trigger = CronTrigger(hour=run_dt.hour, minute=run_dt.minute)
        repeat_label = "daily"
        job_id = f"reminder_daily_{run_dt.hour:02d}{run_dt.minute:02d}"
    elif repeat_rule == "weekly":
        from apscheduler.triggers.cron import CronTrigger

        trigger = CronTrigger(
            day_of_week=run_dt.strftime("%a").lower(),
            hour=run_dt.hour,
            minute=run_dt.minute,
        )
        repeat_label = f"every {run_dt.strftime('%A')}"
        job_id = f"reminder_weekly_{run_dt.strftime('%a').lower()}_{run_dt.hour:02d}{run_dt.minute:02d}"
    else:
        from apscheduler.triggers.date import DateTrigger

        trigger = DateTrigger(run_date=run_dt)
        repeat_label = "once"
        job_id = f"reminder_{run_dt.timestamp():.0f}"

    scheduler.add_job(
        func=_fire_reminder,
        trigger=trigger,
        args=[user_id, message, recipient],
        id=job_id,
        replace_existing=True,
    )
    friendly_time = run_dt.strftime("%A, %d %b %Y at %I:%M %p")
    to_label = recipient or "your email"
    return (
        f"Reminder scheduled ({repeat_label})!\n\n"
        f"Time: {friendly_time}\n"
        f"To: {to_label}\n"
        f"Message: {message}"
    )
