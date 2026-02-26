from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import settings
from langchain_groq import ChatGroq
from src.agents.state import AgentState, IntentClassification, create_empty_travel_state

def get_llm():
    """Get cofigured Groq LLM instance"""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )


INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for a life admin AI assistant.

Classify the user's query into ONE of these intents:

1. **travel_planning**: User wants to plan/book a trip (flights, hotels, weather).
   Examples: "Plan a trip to Pokhara", "Find me a flight to Delhi", "Weekend getaway under 30k"

2. **reminder**: User wants to be reminded or notified about something.
   Examples: "Remind me to call mom at 5pm", "Send me a notification tomorrow"

3. **creative**: User wants creative content (images, ideas, moodboards).
   Examples: "Generate a romantic dinner moodboard", "Create an image of mountains"

4. **unknown**: Query doesn't fit above categories or is unclear.
   Examples: "Hello", "What can you do?", gibberish

Respond ONLY with valid JSON matching this schema:
{{
    "intent": "travel_planning" | "reminder" | "creative"
    "confidence": 0.95,
     "reasoning": "Brief explanation"     
}}"""),
     ("user", "{query}")
])

async def classify_intent_node(state: AgentState) -> AgentState:
    """Classify user intent and initialize appropriate sub-state"""

    query = state["user_query"]

    llm = get_llm()
    chain = INTENT_PROMPT | llm.with_structured_output(IntentClassification)

    try:
        result = await chain.ainvoke({"query": query})
        intent = result.intent
        reasoning = result.reasoning
    except Exception as e:
        print(f"[Orchestrator] Intent classification failed: {e}")
        intent = "unknown"
        reasoning = f"Classification error: {e}"

    # Initilize sub-state
    updates = {
        "intent": intent
    }

    if intent == "travel_planning":
        updates["travel_state"] = create_empty_travel_state
    elif intent == "reminder":
        updates["reminder_state"] ={}

    print(f"[Orchestrator] Classified intent: {intent} ({reasoning})")

    return updates

def route_to_subgraph(state: AgentState) -> Literal["travel_graph", "reminder_graph", "creative_graph", "unknown_handler"]:
    """Route to appropriate sub-graph based on classified intent
    
    This is a conditional edge function - it doesnt modify state but 
    only returns the name of the next node to execute.

    """
    intent = state["intent"]

    routing_map = {
        "travel_planning": "travel_graph",
        "reminder": "reminder_graph",
        "creative": "creative_graph",
        "unknown": "unknown_handler",
    }

    return routing_map.get(intent, "unknown_handler")

async def unknown_handler_node(state: AgentState) -> AgentState:
    """Handle queries that don't fit any intent."""
    return {
        "final_response": (
            "I'm a life admin assistant specialized in:\n\n"
            "• **Travel planning**: Flights, hotels, weather research\n"
            "• **Reminders**: WhatsApp notifications & alerts\n"
            "• **Creative**: Moodboards & AI images\n\n"
            "Could you rephrase your request to match one of these?"
        )
    }
