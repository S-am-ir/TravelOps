"""
Travel agent node.

Primary:  OpenRouter models (no TPM limit, 50 req/day free)
Fallback: Groq (openai/gpt-oss-120b, qwen/qwen3-32b)
"""

import json
import asyncio
from contextvars import ContextVar
from datetime import date
from typing import Optional, Callable, Awaitable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import interrupt
from src.agents.state import AgentState
from src.mcp.client import get_mcp_tools
from src.model_api import get_primary_llm, get_fallback1_llm, get_fallback2_llm

# Contextvar for token streaming callback — set by API layer before invoking graph
token_callback: ContextVar[Optional[Callable[[str], Awaitable[None]]]] = ContextVar(
    "token_callback", default=None
)


TRAVEL_SYSTEM = """You are a WILDLY ENTHUSIASTIC, ultra-knowledgeable, and highly creative travel assistant! 🌍✨
Today is {today}.

You don't just give itineraries—you craft unforgettable, dynamic, and budget-savvy adventures. Your tone should be punchy, highly empathetic, conversational, and packed with emojis. No boring lists. Make the user HYPED for their trip!

━━━ HOW TO DECIDE TOOLS (DO THIS IN PARALLEL) ━━━
CRITICAL: YOU MUST CALL ALL REQUIRED TOOLS AT ONCE IN YOUR FIRST TURN. 
Do NOT search weather, wait for a turn, then search hotels. FIRE THEM ALL IN PARALLEL to save time!
- Planning a trip? → Call `get_weather`, `search_hotels`, and `web_search_multi` SIMULTANEOUSLY.
- Need buses, micro, transport options? → call `web_search`.
- Flight/hotel search failed? → fall back to `web_search` immediately for alternatives.

━━━ DATES ━━━
Calculate relative dates using today's date:
- "this weekend" = upcoming Sat + Sun from today
- "tomorrow" = today + 1

━━━ RESPONSE QUALITY & VIBE ━━━
- **Tone**: Fun, wildly creative, and adventurous. Use bold headings, bullet points, and strategic emojis to make it readable and exciting!
- **Data**: Use REAL data from tools. Never invent prices or locations. If a tool fails, give a best-effort realistic estimate and say so!
- **Budget Plans**: Break down the costs transparently. ALWAYS suggest a sneaky cheap local alternative for food or transport!
- **Formatting**: Make it look beautiful. Ditch the robotic tone.

━━━ SENDING ━━━
Do NOT send telegrams/emails unless the user explicitly asks to send or notify them. Let the Reminder agent handle it.
"""


async def _execute_tool(tc: dict, tool_map: dict) -> ToolMessage:
    name = tc["name"]
    args = tc["args"]
    call_id = tc["id"]

    if name == "send_telegram_message":
        draft = args.get("body", "")
        confirmation = interrupt(
            {
                "type": "telegram_confirmation",
                "draft": draft,
                "to": "your Telegram",
                "prompt": f"Send this to your Telegram?\n\n{draft}\n\nReply yes/no.",
            }
        )
        confirmed = (
            isinstance(confirmation, dict) and confirmation.get("confirmed")
        ) or (
            isinstance(confirmation, str)
            and confirmation.strip().lower() in ("yes", "y", "confirm", "send")
        )
        if not confirmed:
            return ToolMessage(
                content="Cancelled. Message was NOT sent.", tool_call_id=call_id
            )

    tool = tool_map.get(name)
    if not tool:
        return ToolMessage(
            content=f"Tool '{name}' not available. Known: {list(tool_map.keys())}",
            tool_call_id=call_id,
        )

    try:
        result = await tool.ainvoke(args)
        if hasattr(result, "model_dump_json"):
            content = result.model_dump_json()
        elif hasattr(result, "json"):
            content = result.json()
        elif not isinstance(result, str):
            content = json.dumps(result)
        else:
            content = result
        print(f"[Travel] tool OK: {name} → {len(content)} chars")
    except Exception as e:
        content = f"Tool error ({name}): {e}"
        print(f"[Travel] tool FAIL: {name}: {e}")

    return ToolMessage(content=content, tool_call_id=call_id)


async def _react_loop(react_tools, tool_map, system_msg, conversation):
    """ReAct loop with streaming, retry logic, and model fallback."""
    accumulated = []
    MAX_ITER = 6  # Give model enough iterations for sequential tools
    cb = token_callback.get()
    # Groq primary (no retry), Groq fallback (no retry), OpenRouter last resort (retry)
    model_factories = [get_primary_llm, get_fallback1_llm, get_fallback2_llm]
    retry_last_only = {2}  # only retry on OpenRouter (index 2)

    for i in range(MAX_ITER):
        response = None
        for factory_idx, factory in enumerate(model_factories):
            should_retry = factory_idx in retry_last_only
            max_retries = 3 if should_retry else 1
            for retry in range(max_retries):
                try:
                    llm = factory(temperature=0.85).bind_tools(react_tools)
                    parts = []
                    async for chunk in llm.astream(
                        [system_msg] + conversation + accumulated
                    ):
                        parts.append(chunk)
                        if cb and chunk.content:
                            await cb(chunk.content)
                    response = parts[0]
                    for part in parts[1:]:
                        response = response + part
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = any(
                        k in err_str.lower() for k in ["429", "rate", "limit", "quota"]
                    )
                    if is_rate_limit and should_retry and retry < max_retries - 1:
                        wait = 2 ** (retry + 1)
                        print(
                            f"[Travel] iter {i + 1} model {factory_idx} rate-limited, retry {retry + 1}/{max_retries} in {wait}s"
                        )
                        await asyncio.sleep(wait)
                    else:
                        print(
                            f"[Travel] iter {i + 1} model {factory_idx} failed: {type(e).__name__}: {str(e)[:100]}"
                        )
                        break
            if response is not None:
                break
        if response is None:
            print(f"[Travel] all models failed iter {i + 1}, stopping")
            return accumulated, False

        accumulated.append(response)
        print(
            f"[Travel] iter {i + 1}: {len(response.tool_calls)} tool_calls, has_content={bool(response.content)}"
        )

        if not response.tool_calls:
            return accumulated, True

        tool_names = [tc["name"] for tc in response.tool_calls]
        if cb:
            await cb(f"\n*Running: {', '.join(tool_names)}...*\n")

        results = await asyncio.gather(
            *[_execute_tool(tc, tool_map) for tc in response.tool_calls],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"[Travel] gather error: {r}")
            else:
                accumulated.append(r)

    # Force text response if max iterations reached
    if not _last_ai_text(accumulated):
        print("[Travel] max iterations, forcing text response")
        try:
            force_llm = model_factories[0](temperature=0.85)
            force_msgs = (
                [system_msg]
                + conversation
                + accumulated
                + [
                    SystemMessage(
                        content="Respond now using all tool data above. No more tools."
                    )
                ]
            )
            force_resp = await force_llm.ainvoke(force_msgs)
            accumulated.append(force_resp)
        except Exception as e:
            print(f"[Travel] forced response failed: {e}")

    return accumulated, True


async def _fallback_response(conversation):
    """Text-only fallback when all tool-calling models fail."""
    for factory in [get_fallback1_llm, get_fallback2_llm]:
        try:
            llm = factory()
            prompt = (
                "You are a travel assistant. The user asked a travel question but live tools are unavailable. "
                "Give a helpful answer based on your knowledge. Be upfront that info may not be current.\n\n"
                "User: " + (conversation[-1].content if conversation else "")
            )
            r = await llm.ainvoke(prompt)
            text = _extract_text(r.content)
            if text:
                return text
        except Exception as e:
            print(f"[Travel] fallback failed: {str(e)[:80]}")
            continue
    return "I'm having trouble connecting right now. Please try again in a moment."


async def travel_agent_node(state: AgentState) -> dict:
    all_tools = await get_mcp_tools()
    react_tools = [t for t in all_tools if t.name != "send_telegram_message"]
    tool_map = {t.name: t for t in all_tools}

    system_msg = SystemMessage(
        content=TRAVEL_SYSTEM.format(today=date.today().isoformat())
    )
    conversation = list(state.get("messages", []))
    # Filter: only keep HumanMessage and AIMessage (skip ToolMessages from previous turns)
    conversation = [
        m
        for m in conversation
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    ]
    conversation = conversation[-4:] if len(conversation) > 4 else conversation

    msgs, ok = await _react_loop(react_tools, tool_map, system_msg, conversation)
    final = _last_ai_text(msgs)

    if ok and final:
        return {"messages": msgs, "final_response": final}

    print(f"[Travel] react loop done (ok={ok}), trying fallback")
    text = await _fallback_response(conversation + msgs)
    return {"messages": [AIMessage(content=text)], "final_response": text}


def _extract_text(content) -> str:
    import re
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            (
                b.get("text", "")
                if isinstance(b, dict)
                else (b if isinstance(b, str) else "")
            )
            for b in content
        )
    else:
        text = str(content)
    
    # Strip <think>...</think> tags which some OS models leak
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _last_ai_text(messages: list) -> str | None:
    """Return the content of the last AIMessage that has text."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = _extract_text(msg.content)
            if text.strip():
                return text
    return None
