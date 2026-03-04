from typing import Literal
from langchain_core.messages import SystemMessage, AIMessage
from src.agents.state import AgentState, IntentClassification
from src.model_api import invoke_with_fallback


CLASSIFIER_SYSTEM = """You are an intent classifier for a personal life admin AI.

Classify the latest user message into ONE of:

travel_planning — planning or researching trips, flights, hotels, weather, destination info,
  packing advice, itineraries, "what to do in X", budget for travel.

reminder — user wants to be reminded, notified, or alerted. This includes asking to
  send or share information via Telegram, WhatsApp, or any messaging platform.
  CRITICAL: "send me this on Telegram", "notify me via WhatsApp", "send this to me",
  "remind me of this" — even after a travel conversation — is ALWAYS a reminder.

creative — generating images, moodboards, visual concepts, aesthetic exploration.

unknown — greetings, meta questions, anything that doesn't fit above.

IMPORTANT: Look at the FULL conversation history for context.
- A short follow-up like "yes go ahead" or "what about hotels?" inherits the prior intent.
- BUT "send me that on Telegram" or "notify me" after any conversation = reminder.
  The delivery/notification request always overrides previous context.

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
        return {"intent": "unknown"}

    try:
        result, model = await invoke_with_fallback(
            [SystemMessage(content=CLASSIFIER_SYSTEM)] + messages,
            structured_schema=IntentClassification,
        )
        print(f"[Orchestrator] Intent: {result.intent} ({result.confidence:.0%}) via {model} — {result.reasoning}")
        return {"intent": result.intent}
    except Exception as e:
        print(f"[Orchestrator] All classifiers failed: {e}")
        return {"intent": "unknown"}


def route_to_agent(state: AgentState) -> Literal["travel_agent", "reminder_agent", "creative_agent", "unknown_handler"]:
    return {
        "travel_planning": "travel_agent",
        "reminder":        "reminder_agent",
        "creative":        "creative_agent",
        "unknown":         "unknown_handler",
    }.get(state.get("intent", "unknown"), "unknown_handler")


async def unknown_handler_node(state: AgentState) -> dict:
    response = (
        "I'm your personal life admin assistant. Here's what I can help with:\n\n"
        "✈️  **Travel** — flights, hotels, weather, destination tips, itineraries\n"
        "🔔  **Reminders** — send yourself a Telegram reminder at any time\n"
        "🎨  **Creative** — AI moodboards and image generation\n\n"
        "What would you like to do?"
    )
    return {
        "messages": [AIMessage(content=response)],
        "final_response": response,
    }