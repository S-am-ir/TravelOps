import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from src.config.settings import settings
import httpx
from datetime import date, timedelta
from amadeus import Client, ResponseError
import asyncio
import re

mcp = FastMCP("travel", json_response=True)

# MODELS
class WeatherDay(BaseModel):
    date: str
    day_of_week: str
    temp_max_c: float
    temp_min_c: float
    condition: str
    chance_of_rain_pct: int
    chance_of_snow_pct: int

class FlightOffer(BaseModel):
    departure_time: str
    arrival_time: str
    airline: str
    flight_number: str
    price_npr: float
    duration_minutes: int
    direct: bool

class HotelOffer(BaseModel):
    name: str
    price_per_night_npr: float
    rating: Optional[float] = None
    address: str
    amenities: List[str] = Field(default_factory=List)

def _duration_minutes(s: str) -> int:
    h = int(re.search(r"(\d+)H", s).group(1)) if "H" in s else 0
    m = int(re.search(r"(\d+)M", s).group(1)) if "M" in s else 0
    return h * 60 + m

def _dow(iso_date: str) -> str:
    return date.fromisoformat(iso_date).strftime("%A")

def _amadeus() -> Client:
    return Client(
        client_id=settings.amadeus_client_id.get_secret_value(),
        client_secret=settings.amadeus_client_secret.get_secret_value(),
        hostname="production",
    )

@mcp.tool()
async def get_weather(location: str, start_date: str, end_date: Optional[str] = None):
    """Fetch a daily weather forecast for any city using WeatherAPI.
    
    Pass city name directly - no geocoding step needed. Returns up to 14 days
    if forecast. Dates must be YYYY-MM-DD; resolve natural language like "tomorrow"
    or "this weekend" using today's date from the system prompt before calling.
    end_date defaults to next day after start_date is omitted.
    
    Args:
        location: City name e.g. "Pokhara", "Kathmandu", "Namche Bazaar".
        start_date: Forecast start date, YYYY-MM-DD.
        end_date: Forecast end date, YYYY-MM-DD (inclusive) . Optional.
    
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date else start.replace(day=start.day + 1)
    days = (end - start).days + 1
    days = max(1, min(days, 14)) # WeatherAPI free plan: up to 14 days

    async with httpx.AsyncClient(timeout=10) as c:
        resp = await c.get(
            "https://api.weatherapi.com/v1/forecast.json",
            params={
                "key": settings.weatherapi_key.get_secret_value(),
                "q": location,
                "days": days,
                "aqi": "no",
                "alerts": "no",
            },
        )
        data = resp.json()

        if "error" in data:
            return  [{"error": data["error"]["message"]}]
        
        results = []
        for day in data["forecast"]["forecastday"]:
            d = day["day"]
            results.append(WeatherDay(
                date=day["date"],
                day_of_week=_dow(day["date"]),
                temp_max_c=d["maxtemp_c"],
                temp_min_c=d["mintemp_c"],
                condition=d["condition"]["text"],
                chance_of_rain_pct=d["daily_chance_of_rain"],
                chance_of_snow_pct=d["daily_chance_of_snow"],
            ))
        return results
    
@mcp.tool()
async def search_flights(origin: str, destination: str, departure_date: str, adults: int = 1, max_price_npr: float = 20000, return_date: Optional[str] = None) -> List[FlightOffer]:
    """Search for available flights via Amadeus.
    
    Dates must be YYYY-MM-DD. Origin and destination must be IATA airport codes.

    Args:
        origin: IATA airport code e.g. "KTM", "PKR", "DEL".
        destination: IATA airport code e.g. "PKR", "KTM", "DOH".
        departure_date: Outbound flight date, YYYY-MM-DD.
        adults: Number of passengers (default 1).
        max_price_npr:  Max price per person in NPR (default 20000).
        return_date: Return date for round-trip, YYYY-MM-DD. Omit for one-way.
    """
    try:
        params = dict(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=adults,
            currencyCode="NPR",
            max=5,
            maxPrice=int(max_price_npr),
        )
        if return_date:
            params["returnDate"] = return_date

        resp = _amadeus().shopping.flight_offers_search.get(**params)

        offers = []
        for o in resp.data:
            itin = o["itineraries"][0]
            segs = itin["segments"]
            offers.append(FlightOffer(
                departure_time=segs[0]["departure"]["at"],
                arrival_time=segs[-1]["arrival"]["at"],
                airline=segs[0]["carrierCode"],
                flight_number=segs[0]["number"],
                price_npr=float(o["price"]["total"]),
                duration_minutes=_duration_minutes(itin.get("duration", "PT0M")),
                direct=len(segs) == 1,
                stops=len(segs) - 1,
            ))
        return offers or [{"note": "No flights found for these parameters."}]

    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
async def search_hotels(city_code: str, checkin_date: str, checkout_date: str, adults: int=1, max_price_npr: float = 15000,) -> List[HotelOffer]:
    """Search for available hotels via Amadeus. Prices returned in NPR.

    Uses a two-step flow: hotel list by city, then live pricing. Both handled
    internally. Dates must be YYYY-MM-DD. city_code is the IATA city code.

    Args:
        city_code:      IATA city code for destination e.g. KTM, PKR, BHR.
        checkin_date:   Check-in date, YYYY-MM-DD.
        checkout_date:  Check-out date, YYYY-MM-DD. Required.
        adults:         Guests per room (default 1).
        max_price_npr:  Max price per night in NPR (default 15000).
    """
    try:
        client = _amadeus()

        hotel_list = client.reference_data.locations.hotels.by_city.get(cityCode=city_code)
        hotel_ids = [h["hotelId"] for h in hotel_list.data[:30]]
        if not hotel_ids:
            return [{"note": f"No hotels listed for city '{city_code}'."}]

        offers_resp = client.shopping.hotel_offers_search.get(
            hotelIds=",".join(hotel_ids),
            checkInDate=checkin_date,
            checkOutDate=checkout_date,
            adults=adults,
            currencyCode="NPR",
            bestRateOnly=True,
        )

        nights = (date.fromisoformat(checkout_date) - date.fromisoformat(checkin_date)).days
        results = []

        for ho in offers_resp.data:
            hotel = ho["hotel"]
            price_total = float(ho["offers"][0]["price"]["total"])
            price_per_night = round(price_total / max(nights, 1), 2)

            if price_per_night > max_price_npr:
                continue

            addr = hotel.get("address", {})
            results.append(HotelOffer(
                name=hotel["name"],
                price_per_night_npr=price_per_night,
                total_price_npr=price_total,
                rating=str(hotel["rating"]) if hotel.get("rating") else None,
                address=", ".join(addr.get("lines", [])) or addr.get("cityName", city_code),
                amenities=hotel.get("amenities", [])[:6],
            ))

            if len(results) == 5:
                break

        return results or [{"note": "No hotels found within price range."}]

    except Exception as e:
        return [{"error": str(e)}]
    

if __name__ == "__main__":
    print(f"[MCP Travel] running on port {settings.mcp_travel_port}")
    mcp.run(transport="streamable-http")