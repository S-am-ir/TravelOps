from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """Root Graph State"""

    # Conversation history - accumulated across turns via add_message
    messages: Annotated[list, add_messages]

    # Routing - set by classify_intent, re-evaluated each turn
    intent: Literal["travel_planning", "reminder", "unknown"]

    # Final response text (set by terminal node, read by API layer)
    final_response: str

    # Error surfacing
    error: Optional[str]

    # User context (passed from API layer)
    user_id: Optional[str]


class IntentClassification(BaseModel):
    """Structured output from intent classification LLM call."""

    intent: Literal["travel_planning", "reminder", "unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Why this intent was chosen")


class ReminderExtraction(BaseModel):
    """Structured output for reminder intent"""

    reminder_message: str = Field(
        ...,
        description="What the user wants to be reminded about."
        "Be concise - this will be the email body.",
    )

    scheduled_for: str = Field(
        ...,
        description="When to send. ISO datetime (YYYY-MM-DDTHH:MM:SS) if a specific"
        "time was given, else 'now' for immediate send.",
    )
    recipient_email: Optional[str] = Field(
        None,
        description="Email address(es) to send to. If multiple, include ALL as comma-separated string"
        " e.g. 'user1@gmail.com, user2@gmail.com'"
        " Null if not mentioned - will fall back to user's own email.",
    )
    repeat_rule: Optional[Literal["daily", "weekly", "none"]] = Field(
        "none", description="Recurrence. 'none' for one-off reminders."
    )
