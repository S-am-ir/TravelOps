from typing import Literal
from langchain_core.messages import SystemMessage, AIMessage
from src.agents.state import AgentState, IntentClassification
from src.model_api import invoke_with_fallback


CLASSIFIER_SYSTEM = """You are an intent classifier.

Classify the latest user message into ONE of:

travel_planning — planning or researching trips, flights, hotels, weather,
  destination info, packing advice, itineraries, "what to do in X", budget for travel.
  ALSO includes: standalone questions about the weather.

reminder — user wants to be reminded, notified, or alerted via email.
  CRITICAL: "send me an email", "email me", "remind me", "notify me"
  — even after a travel conversation — is ALWAYS a reminder.

general — generic greetings (hi, hello), casual conversation, meta questions (who are you), 
  asking for current date/time, or anything completely unrelated to travel/reminders.

IMPORTANT: Look at the FULL conversation history for context.
- A short follow-up like "yes go ahead" or "what about hotels?" inherits the prior intent.
- BUT "email me that" or "remind me" after any conversation = reminder.

Respond with valid JSON only:
{
    "intent": "travel_planning",
    "confidence": 0.95,
    "reasoning": "User is asking about flights"
}
"""


async def classify_intent_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "general"}

    try:
        result, model = await invoke_with_fallback(
            [SystemMessage(content=CLASSIFIER_SYSTEM)] + messages,
            structured_schema=IntentClassification,
        )
        print(
            f"[Orchestrator] Intent: {result.intent} ({result.confidence:.0%}) via {model} — {result.reasoning}"
        )
        return {"intent": result.intent}
    except Exception as e:
        print(f"[Orchestrator] All classifiers failed: {e}")
        return {"intent": "general"}


def route_to_agent(
    state: AgentState,
) -> Literal["travel_agent", "reminder_agent", "general_agent"]:
    return {
        "travel_planning": "travel_agent",
        "reminder": "reminder_agent",
        "general": "general_agent",
    }.get(state.get("intent", "general"), "general_agent")


async def general_agent_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    now_iso = state.get("user_local_time") or "unknown"
    tz_info = state.get("user_timezone") or "UTC"
    
    system_msg = SystemMessage(
        content="You are a friendly, helpful assistant. The user is asking a general question (such as the date/time), chatting casually, or greeting you. "
                "You do NOT have access to travel search tools or email tools right now. "
                f"Today's local date/time is: {now_iso} ({tz_info}). "
                "Just answer their question naturally or respond to their greeting. Be conversational."
    )
    
    try:
        # Keep recent context
        recent_messages = [m for m in messages if isinstance(m, (SystemMessage, AIMessage, type(messages[0])))] # handle HumanMessage generically
        result, _ = await invoke_with_fallback([system_msg] + recent_messages[-6:])
        response = result.content if hasattr(result, "content") else str(result)
        
        # Strip <think> tags if any
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"[Orchestrator] General agent failed: {e}")
        response = "I'm here! Let me know if you want to plan a trip or need a reminder sent."
        
    return {
        "messages": [AIMessage(content=response)],
        "final_response": response,
    }
