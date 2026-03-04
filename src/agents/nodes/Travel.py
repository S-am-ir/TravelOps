"""
Travel agent node.

LLM strategy:
- PRIMARY:  qwen-qwq-32b on Groq — reasoning model, verified tool calling support,
            handles parallel tool calls cleanly. temperature=0.6, top_p=0.95 per Groq docs.
- FALLBACK: llama-3.3-70b-versatile on Groq — TEXT ONLY, never bind tools.
            Only kicks in if QwQ fails (rate limit, error etc.).
            Groq confirmed: Llama produces broken XML tool calls → 400 BadRequestError.
"""

import json
import asyncio
from datetime import date
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langgraph.types import interrupt
from src.agents.state import AgentState
from src.mcp_internals.client import get_mcp_tools
from src.config.settings import settings


# ── Models ────────────────────────────────────────────────────────────────

QWQ_MODEL   = "qwen/qwen3-32b"
LLAMA_MODEL = "llama-3.3-70b-versatile"

_qwq_base   = None  # qwen/qwen3-32b
_llama_base = None


def _get_qwq(tools=None):
    """qwen/qwen3-32b: primary ReAct loop. Tool use, 128K context, non-thinking mode for speed."""
    global _qwq_base
    from langchain_groq import ChatGroq
    if _qwq_base is None:
        _qwq_base = ChatGroq(
            model=QWQ_MODEL,
            temperature=0.6,
            reasoning_effort="none",   # top-level param — non-thinking mode, fast
            api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
        )
    return _qwq_base.bind_tools(tools) if tools else _qwq_base


def _get_llama():
    """Llama 70B: text-only fallback. NEVER bind tools."""
    global _llama_base
    from langchain_groq import ChatGroq
    if _llama_base is None:
        _llama_base = ChatGroq(
            model=LLAMA_MODEL,
            temperature=0,
            api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
        )
    return _llama_base  # NEVER bind tools — produces broken XML tool calls → Groq 400


# ── Prompts ───────────────────────────────────────────────────────────────

TRAVEL_SYSTEM = """You are a knowledgeable, conversational travel assistant for Nepal.
Today is {today}.

━━━ TOOLS ━━━
get_weather              — daily forecast for any city (use city name, not IATA)
search_flights           — Skyscanner flight search (IATA codes, prices in USD)
search_hotels            — Booking.com hotel search (city name, prices in USD)
search_destination_info  — web search for activities and itinerary ideas

━━━ IATA CODES (Nepal) ━━━
Kathmandu=KTM  Pokhara=PKR  Biratnagar=BIR  Bharatpur=BHR
Nepalgunj=KEP  Lukla=LUA    Bhairahawa=BWA  Dhangadhi=DHI

━━━ DATES ━━━
Today's date is explicitly provided above. ALWAYS use it to calculate:
- "this weekend" = the upcoming Saturday + Sunday from today's date
- "tomorrow" = today + 1 day
- "next Friday" = the next Friday from today
NEVER guess or assume today's date. Use the date provided.

━━━ STRATEGY ━━━
- No dates given? Use the upcoming weekend (Sat-Sun). State this assumption.
- Fire multiple tools in ONE response to avoid slow round-trips.
  e.g. call search_flights + search_hotels + get_weather simultaneously.
- Prices are in USD → always convert to NPR (1 USD = 135 NPR) in your reply.
  "under 15k NPR" is about $111 USD total. Clearly mark which options fit.
- Never invent flight/hotel data — but DO use the fallback estimates the tools
  provide when live data is unavailable (they are clearly labelled as estimates).
- Be concise and conversational. No walls of text.

━━━ HANDLING PARTIAL RESULTS ━━━
The flight/hotel tools return a `status` field:
  "ok"          — real live data, present confidently
  "no_results"  — API worked but no flights on that route/date
  "api_error"   — live search failed; tool may include fallback estimates
  "unavailable" — API key not configured

Rules:
1. If status is "api_error" or "no_results" but flights/hotels list is NON-EMPTY,
   present those estimates clearly labelled as "estimated / typical fares" and
   mention the note field if present.
2. If the list is truly empty AND status is error, say live pricing is unavailable
   and give rough NPR estimates from your knowledge.
3. NEVER refuse to answer just because one tool failed — present what you have
   from the tools that DID succeed (weather ✅, hotels ✅, destination info ✅)
   and handle the failed tool gracefully inline.
4. Do NOT apologise or say "I was unable to retrieve" as the opening line.
   Lead with what you found. Mention limitations inline, briefly.

━━━ TELEGRAM ━━━
Do NOT call send_telegram_message unless the user explicitly asks to send something.
"""

LLAMA_FALLBACK_SYSTEM = """You are a helpful Nepal travel assistant with no live data access.
Today is {today}.

Give a helpful knowledge-based answer. Be upfront that prices are rough estimates.

Nepal travel reference:
- KTM to Pokhara flights: NPR 3,500-8,000 each way (Buddha Air / Yeti Airlines, ~25 min)
- Budget guesthouses Pokhara (Lakeside): NPR 800-2,500/night
- Mid-range hotels: NPR 2,500-6,000/night
- Weekend trip under 15k NPR is very achievable
- Best areas: Phewa Lake / Lakeside for accommodation, Sarangkot for sunrise

Direct user to buddhaair.com or yetiairlines.com for flights, booking.com for hotels.
"""


# ── Tool execution ────────────────────────────────────────────────────────

async def _execute_tool(tc: dict, tool_map: dict) -> ToolMessage:
    name    = tc["name"]
    args    = tc["args"]
    call_id = tc["id"]

    if name == "send_telegram_message":
        draft = args.get("body", "")
        confirmation = interrupt({
            "type":  "telegram_confirmation",
            "draft": draft,
            "to":    "your Telegram",
            "prompt": f"Send this to your Telegram?\n\n{draft}\n\nReply yes/no.",
        })
        confirmed = (
            isinstance(confirmation, dict) and confirmation.get("confirmed")
        ) or (
            isinstance(confirmation, str)
            and confirmation.strip().lower() in ("yes", "y", "confirm", "send")
        )
        if not confirmed:
            return ToolMessage(content="Cancelled. Message was NOT sent.", tool_call_id=call_id)

    tool = tool_map.get(name)
    if not tool:
        return ToolMessage(
            content=f"Tool '{name}' not available. Known: {list(tool_map.keys())}",
            tool_call_id=call_id,
        )

    try:
        result  = await tool.ainvoke(args)
        content = json.dumps(result) if not isinstance(result, str) else result
        print(f"[Travel] OK {name} -> {len(content)} chars")
    except Exception as e:
        content = f"Tool error ({name}): {e}"
        print(f"[Travel] FAIL {name}: {e}")

    return ToolMessage(content=content, tool_call_id=call_id)


# ── QwQ ReAct loop ────────────────────────────────────────────────────────

async def _qwq_loop(react_tools, tool_map, system_msg, conversation):
    accumulated = []
    MAX_ITER = 8

    for i in range(MAX_ITER):
        try:
            llm      = _get_qwq(react_tools)
            response = await llm.ainvoke([system_msg] + conversation + accumulated)
            print(f"[Travel] QwQ iter {i+1}: {len(response.tool_calls)} tool_calls, content={bool(response.content)}")
        except Exception as e:
            print(f"[Travel] QwQ FAILED iter {i+1} ({type(e).__name__}): {e}")
            return accumulated, False

        accumulated.append(response)

        if not response.tool_calls:
            return accumulated, True

        results = await asyncio.gather(
            *[_execute_tool(tc, tool_map) for tc in response.tool_calls],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"[Travel] gather error: {r}")
            else:
                accumulated.append(r)

    return accumulated, True


# ── Llama text-only fallback ──────────────────────────────────────────────

async def _llama_fallback(conversation):
    try:
        llm    = _get_llama()
        system = SystemMessage(content=LLAMA_FALLBACK_SYSTEM.format(today=date.today().isoformat()))
        resp   = await llm.ainvoke([system] + conversation)
        text   = resp.content if hasattr(resp, "content") else str(resp)
        print(f"[Travel] Llama fallback OK ({len(text)} chars)")
        return (
            "_Live search temporarily unavailable — showing general estimates:_\n\n" + text
        )
    except Exception as e:
        print(f"[Travel] Llama fallback FAILED ({type(e).__name__}): {e}")
        return (
            "All providers are unavailable right now. Please try again shortly.\n\n"
            "Quick ref: KTM→Pokhara flights ~NPR 4,000–8,000, "
            "Lakeside guesthouses ~NPR 800–2,500/night. Under 15k for a weekend is very doable."
        )


# ── Main node ─────────────────────────────────────────────────────────────

async def travel_agent_node(state: AgentState) -> dict:
    all_tools   = await get_mcp_tools()
    react_tools = [t for t in all_tools if t.name != "send_telegram_message"]
    tool_map    = {t.name: t for t in all_tools}

    system_msg   = SystemMessage(content=TRAVEL_SYSTEM.format(today=date.today().isoformat()))
    conversation = list(state.get("messages", []))

    # Primary: QwQ-32B ReAct loop with tools
    msgs, ok = await _qwq_loop(react_tools, tool_map, system_msg, conversation)
    final    = _last_ai_text(msgs)
    if ok and final:
        return {"messages": msgs, "final_response": final}

    print(f"[Travel] QwQ done but no final text (ok={ok}), falling back to Llama")
    conversation = conversation + msgs

    # Fallback: Llama 70B text-only
    text = await _llama_fallback(conversation)
    return {"messages": [AIMessage(content=text)], "final_response": text}


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Normalize content blocks (list) or plain string to str."""
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


def _last_ai_text(msgs):
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content:
            return _extract_text(m.content)
    return ""