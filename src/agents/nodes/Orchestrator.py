from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage
from src.config.settings import settings
from src.agents.state import AgentState, IntentClassification


# ── LLM ───────────────────────────────────────────────────────────────────
# qwen3-32b: strong instruction-following, reliable structured output, 1K RPD
# Falls back to llama-3.3-70b-versatile if qwen3 fails for any reason

def get_primary_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        reasoning_effort="none",   # top-level param, NOT model_kwargs
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )

def get_fallback_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )


# ── System Prompt ─────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are an intent classifier for a personal life admin AI.

Classify the latest user message into ONE of:

travel_planning — planning or researching trips, flights, hotels, weather, destination info,
  packing advice, itineraries, "what to do in X", budget for travel.
  Examples: "Plan a trip to Pokhara", "Flights KTM to DEL next Friday",
            "What should I do in Kathmandu for 2 days?", "Is it rainy in Pokhara in March?"

reminder — user wants to be reminded, notified, or alerted. This includes asking to
  send or share information via Telegram, WhatsApp, or any messaging platform.
  CRITICAL: If the user says things like "send me this on Telegram", "notify me via WhatsApp",
  "message me the plan", "send this to me", or "remind me of this" — even immediately after
  a travel conversation — this is ALWAYS a reminder, NOT travel_planning.
  Examples: "Remind me to call mom at 5pm", "Alert me tomorrow morning",
            "Send me a message in 2 hours", "Send me the trip plan on Telegram",
            "Notify me via WhatsApp", "Message me the itinerary"

creative — generating images, moodboards, visual concepts, aesthetic exploration.
  Examples: "Generate a moodboard for a mountain trip", "Create an image of Pokhara at sunset"

unknown — anything that doesn't fit the above, or is a greeting/meta question.
  Examples: "Hi", "What can you do?", "Thanks"

IMPORTANT: Look at the FULL conversation history for context.
- A short follow-up like "yes go ahead" or "what about hotels?" belongs to the same
  intent as the prior messages (travel_planning in that context).
- BUT "send me that on Telegram" or "notify me" after any conversation = reminder.
  The delivery/notification request always overrides the previous context.

Respond with valid JSON only:
{
    "intent": "travel_planning",
    "confidence": 0.95,
    "reasoning": "User is asking about flights"
}
"""

# ── Node ──────────────────────────────────────────────────────────────────

async def classify_intent_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "unknown"}

    llms_to_try = [
        ("qwen3-32b", get_primary_llm),
        ("llama-3.3-70b (fallback)", get_fallback_llm),
    ]

    last_error = None
    for name, llm_factory in llms_to_try:
        try:
            llm = llm_factory()
            structured_llm = llm.with_structured_output(IntentClassification)
            result: IntentClassification = await structured_llm.ainvoke(
                [SystemMessage(content=CLASSIFIER_SYSTEM)] + messages
            )
            print(f"[Orchestrator] Intent: {result.intent} ({result.confidence:.0%}) via {name} — {result.reasoning}")
            return {"intent": result.intent}
        except Exception as e:
            last_error = e
            print(f"[Orchestrator] {name} classification failed: {e}, trying next...")
            continue

    print(f"[Orchestrator] All classifiers failed: {last_error}")
    return {"intent": "unknown"}


# ── Router ────────────────────────────────────────────────────────────────

def route_to_agent(state: AgentState) -> Literal["travel_agent", "reminder_agent", "creative_agent", "unknown_handler"]:
    """Conditional edge: route based on classified intent"""
    return {
        "travel_planning": "travel_agent",
        "reminder":        "reminder_agent",
        "creative":        "creative_agent",
        "unknown":         "unknown_handler",
    }.get(state.get("intent", "unknown"), "unknown_handler")


# ── Unknown handler ───────────────────────────────────────────────────────

async def unknown_handler_node(state: AgentState) -> dict:
    """Friendly fallback for unrecognized queries"""
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