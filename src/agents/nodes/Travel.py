import json
from datetime import date
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langgraph.types import interrupt
from src.agents.state import AgentState
from src.mcp.client  import get_mcp_tools
from src.config.settings import settings
from langchain_groq import ChatGroq

def get_travel_llm():
    """Get configured Groq LLM."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None
    )

TRAVEL_SYSTEM = """You are a knowledgeable, conversational travel assistant for Nepal.
Today is {today}. User's WhatsApp number: {phone}.

━━━ TOOLS ━━━
get_weather          — daily forecast for any city (city name, not IATA)
search_flights       — Amadeus flight search (requires IATA codes)
search_hotels        — Amadeus hotel search (requires IATA city code)
search_destination_info — web search for places, activities, itinerary ideas
send_whatsapp_message   — send a WhatsApp message to the user

━━━ IATA CODES (Examples for NEPAL) ━━━
Kathmandu=KTM, Pokhara=PKR, Biratnagar=BIR, Bharatpur=BHR,
Nepalgunj=KEP, Lukla=LUA, Bhairahawa=BWA, Dhangadhi=DHI

━━━ HOW TO WORK ━━━
- Start with what you have. Don't demand all details upfront — search and present
  options. The user can refine from there.
- If only destination is given (no dates), ask a single clarifying question
  OR proceed with the nearest sensible date range.
- Budget reasoning: present options at multiple price points. Let the user decide
  what to prioritise (e.g. cheap flight + nice hotel, or vice versa).
  Never hard-split budget percentages.
- If the user asks "what to do in X" or "recommend places", call
  search_destination_info — don't make things up.
- Format responses cleanly. Use emoji sparingly. Be conversational, not robotic.

━━━ WHATSAPP RULE ━━━
NEVER call send_whatsapp_message autonomously. Only call it when the user
explicitly asks you to send something to their phone. The system will
automatically pause for confirmation before sending — your job is just to
call the tool with a well-formatted message body.
"""

async def travel_agent_node(state: AgentState) -> dict:
    """
    Reach Loop: LLM reasons -> calls tools -> reasons again -> responds.
    
    interrupt() is used ONLY before send_whatsapp_message - giving user 
    a chance to see the draft and confirm before its sent.
    """
    tools = await get_mcp_tools()
    tool_map = {t.name: t for t in tools}

    llm = get_travel_llm().bind_tools(tools)

    system = SystemMessage(
        content=TRAVEL_SYSTEM.format(
            today=date.today().isoformat(),
            phone=state.get("user_phone") or "not provided",
        )
    )

    # All messages so far (includes prior turns via add_messages reducer)
    conversation = list(state.get("messages", []))
    new_messages: list = []

    MAX_ITERATIONS = 12 # prevents runaway loops

    for iteration in range(MAX_ITERATIONS):
        response = await llm.ainvoke([system] + conversation + new_messages)
        new_messages.append(response)

        # No tool calls -> agent is done for this turn
        if not response.tool_calls:
            break

        # Execute each tool call
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            # HITL gate for WhatsApp
            if tool_name == "send_whatsapp_message":
                draft_body = tool_args.get("body", "")
                to_number = tool_args.get("to_number", state.get("user_phone", ""))

                # Pause graph, surface confirmation request to the API layer.
                # The API layer sends this back to the user and resumes with
                # their yes/no response via Command(resume=...).
                confirmation = interrupt({
                    "type": "whatsapp_confirmation",
                    "to": to_number,
                    "draft": draft_body,
                    "prompt": (
                        f"I'd like to send this to your WhatsApp ({to_number}):\n\n",
                        f"{draft_body}\n\n"
                        "Reply **yes** to confirm or **no** to cancel."
                    ),
                })

                confirmed = (
                    isinstance(confirmation, dict) and confirmation.get("confirmed")
                ) or (
                    isinstance(confirmation, str)
                    and confirmation.strip().lower() in ("yes", "y", "confirm", "send")
                )

                if not confirmed:
                    new_messages.append(
                        ToolMessage(
                            content="User declined. WhatsApp message was not sent.",
                            tool_call_id=tool_call_id,
                        )
                    )
                    continue

            tool = tool_map.get(tool_name)
            if not tool:
                new_messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' not found.",
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            try:
                result = await tool.ainvoke(tool_args)
                content = json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:
                content = f"Tool error: {e}"
                print(f"[Travel] Tool '{tool_name}' failed: {e}")

            new_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call_id)
            )

    # Surface final AI text as final_response for the API layer
    final_text = ""
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content
            break
    
    return {
        "messages": new_messages,
        "final_response": final_text,
    }