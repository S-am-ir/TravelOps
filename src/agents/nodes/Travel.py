from typing import Literal, List, Dict, Any
import asyncio
from datetime import date
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.types import Command, interrupt
from langchain_core.messages import SystemMessage, HumanMessage 
from src.agents.state import TravelState, ExtractedConstraints, ApprovalDecision
from src.agents.utils import (
    parse_natural_date,
    resolve_airport_code,
    calculate_total_cost,
    is_within_budget,
    format_flight_time,
    format_duration,
)
from utils import NEPAL_AIRPORTS
from src.mcp.client import get_mcp_tools
from src.config.settings import settings

def get_llm():
    """Get configured Groq LLM."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None
    )

async def extract_constraints_node(state: TravelState) -> TravelState:
    """Extract travel constraints from query using clean LLM structured output call."""

    query = state.get("user_query", "")
    if not query:
        return {
            "constraints_complete": False,
            "missing_constraints": ["user_query"],
        }
    
    llm = get_llm()
    today = date.today().isoformat()

    # Prepare prompt with schema guidance 
    system_prompt = """You are a precise travel constraint extractor.
Extract the following from the user's query:
- origin: IATA code (e.g., KTM) or city name (e.g., Kathmandu)
- destination: IATA code or city name
- departure_date: YYYY-MM-DD or natural language (e.g., tomorrow, next Friday)
- return_date: YYYY-MM-DD or natural (null if one-way or not mentioned)
- budget_npr: Total budget in NPR (number only, convert k to thousands)
- adults: Number of travelers (default 1 if not mentioned)

List any missing required fields (origin, destination, departure_date, budget_npr) in missing_fields.
If a field is ambiguous or missing, set to null and add to missing_fields if required.

Query: "{query}"
Today: {today} (use for relative dates)"""

    structured_llm = llm.with_structured_output(ExtractedConstraints)
    
    try:
        messages = [
            SystemMessage(content=system_prompt.format(query=query, today=today)),
            HumanMessage(content=query),
        ]
        extracted: ExtractedConstraints = await structured_llm.ainvoke(messages)
        
        if extracted.origin:
            extracted.origin = resolve_airport_code(extracted.origin) or extracted.origin
        if extracted.destination:
            extracted.destination = resolve_airport_code(extracted.destination) or extracted.destination
        if extracted.departure_date:
            extracted.departure_date = parse_natural_date(extracted.departure_date, reference_date=today) or extracted.departure_date
        if extracted.return_date:
            extracted.return_date = parse_natural_date(extracted.return_date, reference_date=today) or extracted.return_date

        print(f"[Travel] LLM extraction complete: {extracted.model_dump()}")

        return {
            "origin": extracted.origin,
            "destination": extracted.destination,
            "departure_date": extracted.departure_date,
            "return_date": extracted.return_date,
            "budget_npr": extracted.budget_npr,
            "adults": extracted.adults,
            "constraints_complete": extracted.is_complete(),
            "missing_constraints": extracted.missing_fields,
        }
    except Exception as e:
        print(f"[Travel] LLM extraction failed: {str(e)}")
        return {
            "constraints_complete": False,
            "missing_constraints": ["origin", "destination", "departure_date", "budget_npr"],
            "errors": state.get("errors", []) + + [f"Constraint extraction error: {str(e)}"],
        }
    
async def validate_budget_node(state: TravelState) -> Command[Literal["research_executor", "constraint_clarifier", "__end__"]]:
    """Early validation: Check if budget is remotely feasible. Hardcoded for now"""
    budget = state.get("budget_npr", 0)
    origin = state.get("origin")
    destination = state.get("destination")

    min_budget = 10000

    if budget < min_budget:
        print(f"[Travel] Budget {budget} below minimum {min_budget}")
        return Command(
            goto="__end__",
            update={
                "budget_feasible": False,
                "errors": [f"Budget too low. Minimum needed: {min_budget}"]
            }
        )
    
    if not state.get("constraints_complete"):
        print(f"[Travel] Missing constraints: {state.get('missing_constraints')}")
        return Command(goto="constraint_clarifier")

    print(f"[Travel] Budget {budget} is feasible, proceeding to research")

    
async def constraint_clarifier_node(state: TravelState) -> TravelState:
    """Ask user for missing information.
    
    In a real CLI, this would prompt interactively, For now it 
    returns an error message listing what's needed.
    """
    missing = state.get("missing_constraints", [])

    field_map = {
        "origin": "departure city/airport",
        "destination": "destination city/ airport",
        "departure_date": "departure_date",
        "budget_npr": "total budget (in NPR)",
    }

    missing_friendly = [field_map.get(f, f) for f in missing]

    return {
        "errors": [
            f"Missing information: {','.join(missing_friendly)}. "
            "Please provide these details to continue."
        ]
    }

async def research_executor_node(state: TravelState) -> TravelState:
    "Execute parallel research using MCP tools"

    tools = await get_mcp_tools()

    # Extract needed params
    origin = state["origin"]
    destination = state["destination"]
    departure_date = state["departure_date"]
    return_date = state.get("return_date")
    budget = state["budget_npr"]
    adults = state.get("adults", 1)


    print(f"[Travel] Starting parallel research: {origin} → {destination} on {departure_date}")
    
    # Prepare tool calls
    tasks = []
    
    # Task 1: Weather forecast
    weather_tool = next((t for t in tools if t.name == "get_weather"), None)
    if weather_tool:
        tasks.append(
            weather_tool.ainvoke({
                "location": destination,
                "start_date": departure_date,
                "end_date": return_date or departure_date,
            })
        )
    else:
        tasks.append(asyncio.sleep(0))  # Placeholder

    flight_tool = next((t for t in tools if t.name == "search_flights"), None)
    if flight_tool:
        tasks.append(
            flight_tool.ainvoke({
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "adults": adults,
                "max_price_npr": budget * 0.6,
                "return_date": return_date,
            })
        )
    else:
        tasks.append(asyncio.sleep(0))

    hotel_tool = next((t for t in tools if t.name == "search_hotels"), None)
    if hotel_tool and return_date:
        # Calculate nights
        dep = date.fromisoformat(departure_date)
        ret = date.fromisoformat(return_date)
        nights = (ret - dep).days
        max_per_night = (budget * 0.4) / max(nights, 1)
        
        # Convert destination to city code (IATA)
        city_code = destination  # Assumes 3-letter code
        
        tasks.append(
            hotel_tool.ainvoke({
                "city_code": city_code,
                "checkin_date": departure_date,
                "checkout_date": return_date,
                "adults": adults,
                "max_price_npr": max_per_night,
            })
        )
    else:
        tasks.append(asyncio.sleep(0))

async def research_executor_node(state: TravelState) -> TravelState:
    """Execute parallel research using MCP tools.
    
    This is where we call external APIs:
    - Weather forecast
    - Flight search
    - Hotel search
    
    We use asyncio.gather for true parallelization.
    """
    tools = await get_mcp_tools()
    
    # Extract needed params
    origin = state["origin"]
    destination = state["destination"]
    departure_date = state["departure_date"]
    return_date = state.get("return_date")
    budget = state["budget_npr"]
    adults = state.get("adults", 1)
    
    print(f"[Travel] Starting parallel research: {origin} → {destination} on {departure_date}")
    
    # Prepare tool calls
    tasks = []
    
    # Task 1: Weather forecast
    weather_tool = next((t for t in tools if t.name == "get_weather"), None)
    if weather_tool:
        tasks.append(
            weather_tool.ainvoke({
                "location": destination,
                "start_date": departure_date,
                "end_date": return_date or departure_date,
            })
        )
    else:
        tasks.append(asyncio.sleep(0))  # Placeholder
    
    # Task 2: Flight search (allocate 60% of budget to flights)
    flight_tool = next((t for t in tools if t.name == "search_flights"), None)
    if flight_tool:
        tasks.append(
            flight_tool.ainvoke({
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "adults": adults,
                "max_price_npr": budget * 0.6,
                "return_date": return_date,
            })
        )
    else:
        tasks.append(asyncio.sleep(0))
    
    # Task 3: Hotel search (allocate 40% of budget to hotels)
    hotel_tool = next((t for t in tools if t.name == "search_hotels"), None)
    if hotel_tool and return_date:
        # Calculate nights
        dep = date.fromisoformat(departure_date)
        ret = date.fromisoformat(return_date)
        nights = (ret - dep).days
        max_per_night = (budget * 0.4) / max(nights, 1)
        
        # Convert destination to city code (IATA)
        city_code = destination  # Assumes 3-letter code
        
        tasks.append(
            hotel_tool.ainvoke({
                "city_code": city_code,
                "checkin_date": departure_date,
                "checkout_date": return_date,
                "adults": adults,
                "max_price_npr": max_per_night,
            })
        )
    else:
        tasks.append(asyncio.sleep(0))
    
    # Execute in parallel with retries
    results = []
    for task in tasks:
        try:
            result = await asyncio.wait_for(task, timeout=30)
            results.append(result)
        except asyncio.TimeoutError:
            results.append({"error": "Request timed out"})
        except Exception as e:
            error_msg = f"{str(e)}"
            results.append({"error": error_msg})
    
    weather_data, flight_data, hotel_data = results
    
    print(f"[Travel] Research complete. Flights: {len(flight_data) if isinstance(flight_data, list) else 0}, "
          f"Hotels: {len(hotel_data) if isinstance(hotel_data, list) else 0}")
    
    return {
        "weather_data": weather_data if isinstance(weather_data, list) else [],
        "flight_options": flight_data if isinstance(flight_data, list) else [],
        "hotel_options": hotel_data if isinstance(hotel_data, list) else [],
    }

async def human_approval_node(state: TravelState) -> TravelState:
    "Present research results and wait for human approval"
    flights = state.get("flight_options", [])
    hotels = state.get("hotel_options", [])
    weather = state.get("weather_date", [])

    prompt_lines = ["Travel Options Found\n"]

    if weather:
        prompt_lines.append("## Weather Forecast")
        for day in weather: 
            prompt_lines.append(
                f"- {day['day_of_week']} {day['date']}: {day['condition']}, "
                f"{day['temp_min_c']}°C - {day['temp_max_c']}°C"
            )
        prompt_lines.append("")

    # Flight options
    if flights:
        prompt_lines.append("## Flight Options")
        for i, flight in enumerate(flights, 1):
            prompt_lines.append(
                f"{i}. {flight['airline']} {flight['flight_number']} - "
                f"{(flight['price_npr'])} - "
                f"{format_duration(flight['duration_minutes'])} - "
                f"{'Direct' if flight['direct'] else f"{flight.get('stops', 0)} stops"}"
            )
        prompt_lines.append("")

    # Hotel options
    if hotels:
        prompt_lines.append("## Hotel Options")
        for i, hotel in enumerate(hotels, 1):
            prompt_lines.append(
                f"{i}. {hotel['name']} - "
                f"{hotel['price_per_night_npr']}/night - "
                f"Rating: {hotel.get('rating', 'N/A')}"
            )
        prompt_lines.append("")

    # Cost estimate
    if flights and hotels:
        cheapest_flight = min(f['price_npr'] for f in flights)
        cheapest_hotel = min(h['price_per_night_npr'] for h in hotels)
        departure = state["departure_date"]
        return_date = state.get("return_date")
        nights = (date.fromisoformat(return_date) - date.fromisoformat(departure)).days if return_date else 1
        
        total = cheapest_flight + (cheapest_hotel * nights)
        budget = state["budget_npr"]
        
        prompt_lines.append(f"## Cost Estimate")
        prompt_lines.append(f"- Cheapest combination: {total}")
        prompt_lines.append(f"- Your budget: {budget}")
        prompt_lines.append(f"- {'✅ Within budget' if total <= budget else '⚠️ Over budget'}")
        prompt_lines.append("")
    
    prompt_lines.append("**Approve to proceed with booking?**")
    
    approval_prompt = "\n".join(prompt_lines)
    
    # Store prompt in state, then interrupt
    print(f"[Travel] Presenting options to user (HITL interrupt)")
    
    # This will pause execution
    approval_input = interrupt(approval_prompt)
    
    # When execution resumes, approval_input will contain user's decision
    # For now, we just store the prompt
    return {
        "approval_prompt": approval_prompt,
    }

async def route_after_approval(state: TravelState) -> Literal["booking_executor", "constraint_clarifier"]:
    """Route based on user's approval decision."""
    if state.get("user_approved"):
        return "booking_executor"
    else:
        return "constraint_clarifier"

async def reminders_node(state: TravelState) -> TravelState:
    """Execute reminder/ notifications through whatsapp"""

    flight_idx = state.get("selected_flight_idx", 0)
    hotel_idx = state.get("selected_hotel_idx", 0)

    flights = state.get("flight_options", [])
    hotels = state.get("hotel_options", [])

    selected_flight = flights[flight_idx] if flights else None
    selected_hotel = hotels[hotel_idx] if hotels else None
    
    # Build summary message
    summary_lines = ["🎉 Your travel plan is ready!\n"]
    
    if selected_flight:
        summary_lines.append(
            f"✈️ Flight: {selected_flight['airline']} {selected_flight['flight_number']}\n"
            f"   Departs: {format_flight_time(selected_flight['departure_time'])}\n"
            f"   Price: {selected_flight['price_npr']}"
        )
    
    if selected_hotel:
        summary_lines.append(
            f"\n🏨 Hotel: {selected_hotel['name']}\n"
            f"   Price: {selected_hotel['price_per_night_npr']}/night"
        )
    
    summary = "\n".join(summary_lines)
    
    # Send WhatsApp notification
    tools = await get_mcp_tools()
    whatsapp_tool = next((t for t in tools if t.name == "send_whatsapp_message"), None)
    
    whatsapp_sent = False
    if whatsapp_tool and state.get("user_phone"):
        try:
            result = await whatsapp_tool.ainvoke({
                "to_number": state["user_phone"],
                "body": summary,
            })
            whatsapp_sent = result.get("status") == "sent"
            print(f"[Travel] WhatsApp sent: {whatsapp_sent}")
        except Exception as e:
            print(f"[Travel] WhatsApp failed: {e}")
    
    return {
        "whatsapp_sent": whatsapp_sent,
    }