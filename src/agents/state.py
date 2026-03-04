from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentState(TypedDict, total=False):
    """Root Graph State"""
    # Conversation history - accumulated across turns via add_message
    messages: Annotated[list, add_messages]

    # Routing - set by classfy_intent, re-evaluated each turn
    intent: Literal["travel_planning", "reminder", "creative", "unknown"]

    # Final response text (set by terminal node, read by API layer)
    final_response: str
    
    # Error surfacing
    error: Optional[str]

class IntentClassification(BaseModel):
    """Structured output from intent classification LLM call."""

    intent: Literal["travel_planning", "reminder", "creative", "unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Why this intent was chosen")

class ReminderExtraction(BaseModel):
    """Structured output for reminder intent"""
    reminder_message: str = Field(
        ..., 
        description = "What the user wants to be reminded about."
                      "Be concise - this will be the Whatsapp message body."
    )

    scheduled_for: str = Field(
        ...,
        description="When to send. ISO datetime (YYYY-MM-DDTHH:MM:SS) if a specific"
        "time was given, else 'now' for immediate send.",
    )
    to_number: Optional[str] = Field(
        None,
        description="Phone number override if user specified one (internation format , no +)."
        "Null if not mentioned - will fall back to user_phone from context."
    )
    repeat_rule: Optional[Literal["daily", "weekly", "none"]] = Field(
        "none",
        description="Recurrence. 'none' for one-off reminders."
    )

class CreativeExtraction(BaseModel):
    """Structured output for creative / moodboard intent."""

    visual_prompt: str = Field(
        ...,
        description="Rich, cinematically descriptive prompt ready for image generation."
        "Expand the user's request into lighting, mood, setting, style details."
    )  
    count: int = Field(
        1,
        ge=1,
        le=2,
        description="Number of images to generate. Default 1."
    )

