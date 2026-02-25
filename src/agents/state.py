from typing import TypedDict, Literal, Optional, List, Dict, Any
from datetime import date
from pydantic import BaseModel, Field

class AgentState(TypedDict, total=False):
    """Root State for the Orchestrator"""

    # Input
    user_query: str
    user_phone: Optional[str] # For WhatsApp notifications

    # Routing 
    intent: Literal["travel_planning", "reminder", "creative", "unknown"]

    # Sub-graph states (each intent has its own)
    travel_state: Optional["TravelState"]
    reminder_state: Optional[Dict[str, Any]]

    # Output
    final_response: str
    error: Optional[str]

class TravelState(TypedDict, total=False):
    """State for travel planning sub-graph."""

    # Input (extracted from user query)
    origin: Optional[str]
    destination: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    budget_npr: Optional[float]
    adults: int

    # Extracted constraints
    constraints_complete: bool
    missing_constraints: List[str]

    # Research results (from MCP tools)
    weather_data: Optional[List[Dict[str, Any]]]
    flight_options: Optional[List[Dict[str, Any]]]
    hotel_options: Optional[List[Dict[str, Any]]]

    # Budget tracking
    estimated_cost_npr: float
    budget_feasible: bool

    # Human apporval
    approval_prompt: Optional[str]
    user_approved: Optional[bool]
    selected_flight_idx: Optional[int]
    selected_hotel_idx: Optional[int]

    # Reminder / notifications
    whatsapp_sent: bool

    # Moodboard
    moodboard_images: Optional[List[str]]

    # Error tracking
    errors: List[str]
    retry_count: int

class ExtractedConstraints(BaseModel):
    """Structured output from constraint extraction LLM call."""

    origin: Optional[str] = Field(None, description="IATA airport code e.g. KTM, PKR")
    destination: Optional[str] = Field(None, description="IATA airport code")
    departure_date: Optional[str] = Field(None, description="YYYY-MM-DD format")
    return_date: Optional[str] = Field(None, description="YYYY-MM-DD, null for one-way")
    budget_npr: Optional[float] = Field(None, description="Total budget in NPR")
    adults: int = Field(1, description="Number of travelers")

    missing_fields: List[str] = Field(
        default_factory=list,
        description="Fields that couldn't be extracted from query"
    )

    def is_complete(self) -> bool:
        """Check if we have minimum required constraints"""
        required = [self.origin, self.destination, self.departure_date, self.budget_npr]
        return all(required) and not self.missing_fields

class IntentClassification(BaseModel):
    """Structured output from intent classification LLM call."""

    intent: Literal["travel_planning", "reminder", "creative", "unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Why this intent was chosen")

class ApprovalDecision(BaseModel):
    """User's approval decision from HITL"""

    approved: bool
    selected_flight_idx: Optional[int] = None
    selected_hotel_idx: Optional[int] = None
    feedback: Optional[str] = None # Feedback from the user

def create_empty_travel_state() -> TravelState:
    """Initialize empty travel state with defaults"""
    return TravelState(
        adults=1,
        constraints_complete=False,
        missing_constraints=[],
        estimated_cost_npr=0.0,
        budget_feasible=False,
        whatsapp_sent=False,
        errors=[],
        retry_count=0,
    )