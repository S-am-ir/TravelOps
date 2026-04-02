from typing import Literal
from langchain_core.messages import SystemMessage, AIMessage
from src.agents.state import AgentState, IntentClassification
from src.model_api import invoke_with_fallback


CLASSIFIER_SYSTEM = """You are an intent classifier.

Classify the latest user message into ONE of:

travel_planning — planning or researching trips, flights, hotels, weather,
  destination info, packing advice, itineraries, "what to do in X", budget for travel.

reminder — user wants to be reminded, notified, or alerted via email.
  CRITICAL: "send me an email", "email me", "remind me", "notify me"
  — even after a travel conversation — is ALWAYS a reminder.

unknown — greetings, meta questions, anything that doesn't fit above.

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
        return {"intent": "unknown"}

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
        return {"intent": "unknown"}


def route_to_agent(
    state: AgentState,
) -> Literal["travel_agent", "reminder_agent", "unknown_handler"]:
    return {
        "travel_planning": "travel_agent",
        "reminder": "reminder_agent",
        "unknown": "unknown_handler",
    }.get(state.get("intent", "unknown"), "unknown_handler")


async def unknown_handler_node(state: AgentState) -> dict:
    response = (
        "Here's what I can help with:\n\n"
        "✈️  **Travel** — flights, hotels, weather, destination tips, itineraries\n"
        "📧  **Reminders** — email yourself or others a reminder at any time\n\n"
        "What would you like to do?"
    )
    return {
        "messages": [AIMessage(content=response)],
        "final_response": response,
    }
