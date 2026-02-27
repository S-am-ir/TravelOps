from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Any
import re
import dateparser

def parse_natural_date(text: str, reference_date: Optional[date] = None, languages: List[str] = ["en"]) -> Optional[str]:
    """Converts natural language / fuzzy date strings to YYYY-MM-DD
    Examples:
        "tomorrow"              → "2026-02-25"
        "next friday"           → "2026-02-28"

    """
    if not text:
        return 
    
    today = reference_date or date.today()
    ref_datetime = datetime.combine(today, datetime.min.time())

    parsed = dateparser.parse(
        text,
        languages=languages,
        settings={
            "DATE_ORDER": "DMY", # Nepal/Asia preference
            "PREFER_LOCALE_DATE_ORDER": False,
            "RELATIVE_BASE": ref_datetime,
            "STRICT_PARSING": False,
            "RETURN_AS_TIMEZONE_AWARE": False,
            "PREFER_DAY_OF_MONTH": "current",
        }
    )

    if parsed:
        return parsed.date().isoformat()
    
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text.strip()):
        try:
            date.fromisoformat(text.strip())
            return text.strip()
        except ValueError:
            pass
    
    return None

def calculate_total_cost(flight_price: float, hotel_price_per_night: float, nights: int,) -> Dict[str, float]:
    """Calculate itemized cost breakdown (NPR)"""
    hotel_total = hotel_price_per_night * nights
    return {
        "flight": round(flight_price, 2),
        "hotel": round(hotel_total, 2),
        "total": round(flight_price + hotel_total, 2),
    }

def is_within_budget(total_cost: float, budget: float, buffer_pct: float = 0.95) -> bool:
    """Check if cost is within budget with safety buffer"""
    if budget <= 0:
        return False
    return total_cost <= (budget * buffer_pct)


NEPAL_AIRPORTS = {
    "kathmandu": "KTM",
    "pokhara": "PKR",
    "biratnagar": "BIR",
    "bharatpur": "BHR",
    "nepalgunj": "KEP",
    "lukla": "LUA",
    "tumlingtar": "TMI",
    "janakpur": "JKR",
    "simara": "SIF",
    "bhairahawa": "BWA", 
    "siddharthanagar": "BWA",
    "dhangadhi": "DHI",
    "surkhet": "SKH",
    "jomsom": "JMO",
}

def resolve_airport_code(location: str) -> Optional[str]:
    """Convert city name (or IATA) → IATA code. Nepal-focused.

    Returns None if unknown → caller should handle gracefully
    (e.g. ask user to clarify / pick from list / fallback).

    Examples:
        "Kathmandu"  → "KTM"
        "ktm"        → "KTM"
        "LUA"        → "LUA" (passthrough if 3 uppercase letters)
        "New York"   → None
    """
    if not location:
        return None

    loc = location.lower().strip()

    # Already looks like IATA code
    if len(loc) == 3 and loc.isupper() and loc.isalpha():
        return loc

    # City name lookup (normalized)
    return NEPAL_AIRPORTS.get(loc)

def format_flight_time(iso_datetime: str) -> str:
    """Format ISO datetime to human-readable time.
    
    Example: "2026-03-15T07:30:00" → "07:30 AM"
    """
    try:
        dt = datetime.fromisoformat(iso_datetime.replace("Z", "+00:00"))
        return dt.strftime("%I:%M %p")
    except:
        return iso_datetime
    
def format_duration(minutes: int) -> str:
    """Format flight duration.
    
    Example: 195 → "3h 15m"
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m" if mins else f"{hours}h"