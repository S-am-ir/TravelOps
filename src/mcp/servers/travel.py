import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import Optional, List, Union
from src.config.settings import settings
import httpx

mcp = FastMCP(
    "travel",
    host=settings.mcp_host,
    port=settings.mcp_travel_port,
    json_response=True,
)


# ── Weather ───────────────────────────────────────────────────────────────


class DayForecast(BaseModel):
    date: str
    condition: str
    temp_max_c: float
    temp_min_c: float
    rain_chance_pct: int


class WeatherResult(BaseModel):
    city: str
    forecast: List[DayForecast]
    error: Optional[str] = None


@mcp.tool()
async def get_weather(city: str, days: Union[int, str] = 3) -> WeatherResult:
    """Get weather forecast for a city (use city name, not IATA code).

    Args:
        city: City name e.g. "Pokhara", "Kathmandu". Do not pass a days argument.

    Returns:
        WeatherResult with daily forecasts or error.
    """
    if not settings.weatherapi_key:
        return WeatherResult(city=city, forecast=[], error="WEATHERAPI_KEY not set")

    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 3

    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": settings.weatherapi_key.get_secret_value(),
        "q": city,
        "days": max(1, min(days, 7)),
    }

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return WeatherResult(
                city=city,
                forecast=[],
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            return WeatherResult(city=city, forecast=[], error=str(e))

    forecasts = []
    for day in data.get("forecast", {}).get("forecastday", []):
        forecasts.append(
            DayForecast(
                date=day["date"],
                condition=day["day"]["condition"]["text"],
                temp_max_c=day["day"]["maxtemp_c"],
                temp_min_c=day["day"]["mintemp_c"],
                rain_chance_pct=int(day["day"].get("daily_chance_of_rain", 0)),
            )
        )

    return WeatherResult(city=data["location"]["name"], forecast=forecasts)


# ── Flights ───────────────────────────────────────────────────────────────


class FlightOption(BaseModel):
    airline: str
    departure: str
    arrival: str
    duration_minutes: int
    price_usd: float
    price_npr: int


class FlightResult(BaseModel):
    origin: str
    destination: str
    date: str
    flights: List[FlightOption]
    # Clear status field so the LLM can distinguish "no results" from "API error"
    status: str  # "ok" | "no_results" | "api_error" | "unavailable"
    note: Optional[str] = None


_ENTITY_IDS = {
    "KTM": "95673458",
    "PKR": "128667758",
    "PKRA": "27545974",
    "BIR": "95673497",
    "BHR": "95673502",
    "KEP": "95673510",
    "LUA": "95673516",
    "BWA": "95673522",
    "DHI": "95673528",
}


async def _skyscrapper_search(
    origin: str,
    destination: str,
    date_str: str,
    rapidapi_key: str,
) -> FlightResult:
    """Attempt Sky-Scrapper search. Returns FlightResult with status set."""

    origin_entity = _ENTITY_IDS.get(origin.upper())
    destination_entity = _ENTITY_IDS.get(destination.upper())

    if not origin_entity or not destination_entity:
        # Try to look up unknown airports
        origin_entity, destination_entity = await _lookup_entities(
            origin, destination, rapidapi_key
        )

    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "sky-scrapper.p.rapidapi.com",
    }
    params = {
        "originSkyId": origin.upper(),
        "destinationSkyId": destination.upper(),
        "originEntityId": origin_entity or "",
        "destinationEntityId": destination_entity or "",
        "date": date_str,
        "adults": "1",
        "cabinClass": "economy",
        "currency": "USD",
        "market": "en-US",
        "countryCode": "NP",
        "sortBy": "best",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(
                "https://sky-scrapper.p.rapidapi.com/api/v2/flights/searchFlightsWebComplete",
                headers=headers,
                params=params,
            )
        except Exception as e:
            return FlightResult(
                origin=origin,
                destination=destination,
                date=date_str,
                flights=[],
                status="api_error",
                note=f"Network error: {e}",
            )

    if resp.status_code != 200:
        return FlightResult(
            origin=origin,
            destination=destination,
            date=date_str,
            flights=[],
            status="api_error",
            note=f"HTTP {resp.status_code}",
        )

    data = resp.json()

    # Sky-Scrapper returns status=False on their side errors
    if not data.get("status", True):
        return FlightResult(
            origin=origin,
            destination=destination,
            date=date_str,
            flights=[],
            status="api_error",
            note=f"Sky-Scrapper error: {data.get('message', 'unknown')}",
        )

    itins = (data.get("data") or {}).get("itineraries") or []
    if not itins:
        return FlightResult(
            origin=origin,
            destination=destination,
            date=date_str,
            flights=[],
            status="no_results",
            note="No itineraries returned for this route/date.",
        )

    flights = []
    for itin in itins[:5]:
        try:
            leg = itin["legs"][0]
            segment = leg["segments"][0]
            price = itin["price"]["raw"]
            price_npr = int(price * 135)
            flights.append(
                FlightOption(
                    airline=segment.get("marketingCarrier", {}).get("name", "Unknown"),
                    departure=leg.get("departure", ""),
                    arrival=leg.get("arrival", ""),
                    duration_minutes=leg.get("durationInMinutes", 0),
                    price_usd=round(price, 2),
                    price_npr=price_npr,
                )
            )
        except (KeyError, IndexError, TypeError):
            continue

    return FlightResult(
        origin=origin,
        destination=destination,
        date=date_str,
        flights=flights,
        status="ok" if flights else "no_results",
    )


async def _lookup_entities(origin: str, destination: str, rapidapi_key: str):
    """Look up Sky-Scrapper entityIds for unknown airport codes."""
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "sky-scrapper.p.rapidapi.com",
    }
    origin_entity = None
    destination_entity = None

    async with httpx.AsyncClient(timeout=10) as client:
        for code, store_var in [(origin, "origin"), (destination, "destination")]:
            try:
                r = await client.get(
                    "https://sky-scrapper.p.rapidapi.com/api/v1/flights/searchAirport",
                    headers=headers,
                    params={"query": code, "locale": "en-US"},
                )
                if r.status_code == 200:
                    results = r.json().get("data", [])
                    if results:
                        entity_id = results[0].get("entityId")
                        if store_var == "origin":
                            origin_entity = entity_id
                        else:
                            destination_entity = entity_id
            except Exception:
                pass

    return origin_entity, destination_entity


# Nepal domestic fallback data (used when Sky-Scrapper fails)
_NEPAL_DOMESTIC_FALLBACK = {
    ("KTM", "PKR"): [
        FlightOption(
            airline="Buddha Air",
            departure="07:00",
            arrival="07:25",
            duration_minutes=25,
            price_usd=29.0,
            price_npr=3915,
        ),
        FlightOption(
            airline="Yeti Airlines",
            departure="09:30",
            arrival="09:55",
            duration_minutes=25,
            price_usd=33.0,
            price_npr=4455,
        ),
        FlightOption(
            airline="Shree Airlines",
            departure="14:00",
            arrival="14:25",
            duration_minutes=25,
            price_usd=37.0,
            price_npr=4995,
        ),
    ],
    ("PKR", "KTM"): [
        FlightOption(
            airline="Buddha Air",
            departure="08:00",
            arrival="08:25",
            duration_minutes=25,
            price_usd=29.0,
            price_npr=3915,
        ),
        FlightOption(
            airline="Yeti Airlines",
            departure="10:30",
            arrival="10:55",
            duration_minutes=25,
            price_usd=33.0,
            price_npr=4455,
        ),
    ],
}


@mcp.tool()
async def search_flights(
    origin: str,
    destination: str,
    date: str,
    adults: Union[int, str] = 1,
) -> FlightResult:
    """Search for flights between two airports.

    Args:
        origin:      IATA code e.g. "KTM"
        destination: IATA code e.g. "PKR"
        date:        Travel date YYYY-MM-DD
        adults:      Number of passengers (default 1)

    Returns:
        FlightResult. Check `status`:
          "ok"          — real results found, use `flights` list
          "no_results"  — API worked but no flights for this route/date
          "api_error"   — API failed; fallback estimates provided in `flights`
          "unavailable" — no API key configured
    """
    if not settings.rapidapi_key:
        return FlightResult(
            origin=origin,
            destination=destination,
            date=date,
            flights=[],
            status="unavailable",
            note="RAPIDAPI_KEY not configured",
        )

    # Coerce adults to int — LLMs sometimes pass "1" as string
    try:
        adults = int(adults)
    except (TypeError, ValueError):
        adults = 1

    result = await _skyscrapper_search(
        origin=origin,
        destination=destination,
        date_str=date,
        rapidapi_key=settings.rapidapi_key.get_secret_value(),
    )

    # If Sky-Scrapper failed, inject fallback estimates for known Nepal routes
    if result.status in ("api_error", "no_results") and not result.flights:
        key = (origin.upper(), destination.upper())
        fallback = _NEPAL_DOMESTIC_FALLBACK.get(key)
        if fallback:
            result.flights = fallback
            result.note = (
                f"Live search unavailable ({result.note}). "
                "Showing typical Nepal domestic fares — book at buddhaair.com or yetiairlines.com."
            )

    return result


# ── Hotels ────────────────────────────────────────────────────────────────


class HotelOption(BaseModel):
    name: str
    stars: Optional[float] = None
    review_score: Optional[float] = None
    price_per_night_usd: float
    price_per_night_npr: int
    checkin: str
    checkout: str


class HotelResult(BaseModel):
    city: str
    checkin: str
    checkout: str
    hotels: List[HotelOption]
    status: str  # "ok" | "no_results" | "api_error" | "unavailable"
    note: Optional[str] = None


@mcp.tool()
async def search_hotels(
    city: str,
    checkin: str,
    checkout: str,
    adults: Union[int, str] = 1,
) -> HotelResult:
    """Search for hotels in a city.

    Args:
        city:     City name e.g. "Pokhara", "Kathmandu"
        checkin:  YYYY-MM-DD
        checkout: YYYY-MM-DD
        adults:   Number of guests (default 1)

    Returns:
        HotelResult with a list of options sorted by price,
        or status "api_error"/"unavailable" with note explaining why.
    """
    if not settings.rapidapi_key:
        return HotelResult(
            city=city,
            checkin=checkin,
            checkout=checkout,
            hotels=[],
            status="unavailable",
            note="RAPIDAPI_KEY not configured",
        )

    # Coerce adults to int — LLMs sometimes pass "1" as string
    try:
        adults = int(adults)
    except (TypeError, ValueError):
        adults = 1

    headers = {
        "X-RapidAPI-Key": settings.rapidapi_key.get_secret_value(),
        "X-RapidAPI-Host": "booking-com15.p.rapidapi.com",
    }

    # Step 1: resolve city to dest_id
    dest_id = None
    dest_type = "city"
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(
                "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination",
                headers=headers,
                params={"query": f"{city} Nepal", "languagecode": "en-us"},
            )
            if r.status_code == 200:
                dests = r.json().get("data", [])
                # Prefer city-type result
                for d in dests:
                    if d.get("dest_type") == "city":
                        dest_id = d["dest_id"]
                        dest_type = d["dest_type"]
                        break
                if not dest_id and dests:
                    dest_id = dests[0]["dest_id"]
                    dest_type = dests[0].get("dest_type", "city")
        except Exception as e:
            return HotelResult(
                city=city,
                checkin=checkin,
                checkout=checkout,
                hotels=[],
                status="api_error",
                note=f"Destination lookup failed: {e}",
            )

    if not dest_id:
        return HotelResult(
            city=city,
            checkin=checkin,
            checkout=checkout,
            hotels=[],
            status="no_results",
            note=f"No destination found for '{city}'",
        )

    # Step 2: search hotels
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(
                "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels",
                headers=headers,
                params={
                    "dest_id": dest_id,
                    "search_type": dest_type,
                    "arrival_date": checkin,
                    "departure_date": checkout,
                    "adults": str(adults),
                    "room_qty": "1",
                    "languagecode": "en-us",
                    "currency_code": "USD",
                    "page_number": "1",
                },
            )
        except Exception as e:
            return HotelResult(
                city=city,
                checkin=checkin,
                checkout=checkout,
                hotels=[],
                status="api_error",
                note=f"Hotel search failed: {e}",
            )

    if r.status_code != 200:
        return HotelResult(
            city=city,
            checkin=checkin,
            checkout=checkout,
            hotels=[],
            status="api_error",
            note=f"HTTP {r.status_code}: {r.text[:200]}",
        )

    raw_hotels = r.json().get("data", {}).get("hotels", [])
    if not raw_hotels:
        return HotelResult(
            city=city,
            checkin=checkin,
            checkout=checkout,
            hotels=[],
            status="no_results",
        )

    options = []
    for h in raw_hotels[:8]:
        prop = h.get("property", {})
        price = prop.get("priceBreakdown", {}).get("grossPrice", {}).get("value")
        if price is None:
            continue
        price_usd = round(float(price), 2)
        options.append(
            HotelOption(
                name=prop.get("name", "Unknown"),
                stars=prop.get("propertyClass"),
                review_score=prop.get("reviewScore"),
                price_per_night_usd=price_usd,
                price_per_night_npr=int(price_usd * 135),
                checkin=checkin,
                checkout=checkout,
            )
        )

    # Sort cheapest first
    options.sort(key=lambda x: x.price_per_night_usd)

    return HotelResult(
        city=city,
        checkin=checkin,
        checkout=checkout,
        hotels=options,
        status="ok" if options else "no_results",
    )


if __name__ == "__main__":
    print(f"[MCP Travel] running on port {settings.mcp_travel_port}")
    mcp.run(transport="streamable-http")
