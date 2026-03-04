"""
Travel agent node.

Primary:  qwen/qwen3-32b  (Groq) — ReAct loop with parallel tool calls.
Fallback: llama-3.3-70b-versatile (Groq) — text-only, tools never bound.
          Groq confirmed Llama produces broken XML tool calls → 400 errors.
"""

import json
import asyncio
from datetime import date
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langgraph.types import interrupt
from src.agents.state import AgentState
from src.mcp_internals.client import get_mcp_tools
from src.model_api import get_primary_llm, get_fallback_llm


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
NEVER guess or assume today's date.

━━━ STRATEGY ━━━
- No dates given? Use the upcoming weekend (Sat-Sun). State this assumption.
- Fire multiple tools in ONE response to avoid slow round-trips.
- Prices are USD → convert to NPR (1 USD = 135 NPR). Mark which options fit budget.
- Never invent flight/hotel data — use the fallback estimates tools provide (clearly labelled).

━━━ HANDLING PARTIAL RESULTS ━━━
Tool `status` field values:
  "ok"          — real live data
  "no_results"  — API worked but no flights/hotels for that route/date
  "api_error"   — live search failed; tool may include fallback estimates
  "unavailable" — API key not configured

Rules:
1. If status is "api_error"/"no_results" but flights/hotels list is NON-EMPTY,
   present those estimates labelled as "estimated / typical fares" with the note field.
2. If list is truly empty AND status is error, give rough NPR estimates from knowledge.
3. NEVER refuse because one tool failed — lead with what succeeded, handle failures inline.
4. Do NOT open with "I was unable to retrieve". Lead with what you found.

━━━ TELEGRAM ━━━
Do NOT call send_telegram_message unless the user explicitly asks to send something.
"""

LLAMA_FALLBACK_SYSTEM = """You are a helpful Nepal travel assistant with no live data access.
Today is {today}.

Give a helpful knowledge-based answer. Be upfront that prices are rough estimates.

Nepal travel reference:
- KTM to Pokhara flights: NPR 3,500–8,000 each way (Buddha Air / Yeti Airlines, ~25 min)
- Budget guesthouses Pokhara (Lakeside): NPR 800–2,500/night
- Mid-range hotels: NPR 2,500–6,000/night
- Weekend trip under 15k NPR is very achievable
- Best areas: Phewa Lake / Lakeside for accommodation, Sarangkot for sunrise

Direct user to buddhaair.com or yetiairlines.com for flights, booking.com for hotels.
"""


async def _execute_tool(tc: dict, tool_map: dict) -> ToolMessage:
    name    = tc["name"]
    args    = tc["args"]
    call_id = tc["id"]

    if name == "send_telegram_message":
        draft = args.get("body", "")
        confirmation = interrupt({
            "type":   "telegram_confirmation",
            "draft":  draft,
            "to":     "your Telegram",
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
        print(f"[Travel] tool OK: {name} → {len(content)} chars")
    except Exception as e:
        content = f"Tool error ({name}): {e}"
        print(f"[Travel] tool FAIL: {name}: {e}")

    return ToolMessage(content=content, tool_call_id=call_id)


async def _qwq_loop(react_tools, tool_map, system_msg, conversation):
    """Primary ReAct loop using qwen3-32b with tool calling."""
    accumulated = []
    MAX_ITER    = 8

    for i in range(MAX_ITER):
        try:
            llm      = get_primary_llm(temperature=0.6).bind_tools(react_tools)
            response = await llm.ainvoke([system_msg] + conversation + accumulated)
            print(f"[Travel] iter {i+1}: {len(response.tool_calls)} tool_calls, has_content={bool(response.content)}")
        except Exception as e:
            print(f"[Travel] qwen3 failed iter {i+1} ({type(e).__name__}): {e}")
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


async def _llama_fallback(conversation):
    """Text-only fallback — no tools bound."""
    try:
        llm    = get_fallback_llm()
        system = SystemMessage(content=LLAMA_FALLBACK_SYSTEM.format(today=date.today().isoformat()))
        resp   = await llm.ainvoke([system] + conversation)
        text   = resp.content if hasattr(resp, "content") else str(resp)
        print(f"[Travel] Llama fallback OK ({len(text)} chars)")
        return "_Live search temporarily unavailable — showing general estimates:_\n\n" + text
    except Exception as e:
        print(f"[Travel] Llama fallback failed ({type(e).__name__}): {e}")
        return (
            "All providers are unavailable right now. Please try again shortly.\n\n"
            "Quick ref: KTM→Pokhara flights ~NPR 4,000–8,000, "
            "Lakeside guesthouses ~NPR 800–2,500/night. Under 15k for a weekend is very doable."
        )


async def travel_agent_node(state: AgentState) -> dict:
    all_tools   = await get_mcp_tools()
    react_tools = [t for t in all_tools if t.name != "send_telegram_message"]
    tool_map    = {t.name: t for t in all_tools}

    system_msg   = SystemMessage(content=TRAVEL_SYSTEM.format(today=date.today().isoformat()))
    conversation = list(state.get("messages", []))

    msgs, ok = await _qwq_loop(react_tools, tool_map, system_msg, conversation)
    final    = _last_ai_text(msgs)

    if ok and final:
        return {"messages": msgs, "final_response": final}

    print(f"[Travel] qwen3 done but no final text (ok={ok}), falling back to Llama")
    text = await _llama_fallback(conversation + msgs)
    return {"messages": [AIMessage(content=text)], "final_response": text}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b["text"] if isinstance(b, dict) and b.get("type") == "text" else (b if isinstance(b, str) else "")
            for b in content
        )
    return str(content)


def _last_ai_text(msgs):
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content:
            return _extract_text(m.content)
    return ""