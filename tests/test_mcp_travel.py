"""Tests for the MCP travel server tools — using mocks for API calls."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestGetWeather:
    @pytest.mark.asyncio
    @patch("src.mcp.servers.travel.httpx.AsyncClient")
    async def test_weather_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "location": {"name": "Kathmandu"},
            "forecast": {
                "forecastday": [
                    {
                        "date": "2026-04-01",
                        "day": {
                            "condition": {"text": "Sunny"},
                            "maxtemp_c": 25.0,
                            "mintemp_c": 12.0,
                            "daily_chance_of_rain": 10,
                        },
                    }
                ]
            },
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_cls.return_value = mock_client

        from src.mcp.servers.travel import get_weather
        from src.config.settings import settings

        settings.weatherapi_key = MagicMock(get_secret_value=lambda: "test_key")

        result = await get_weather("Kathmandu", 3)
        assert result.city == "Kathmandu"
        assert len(result.forecast) == 1
        assert result.forecast[0].condition == "Sunny"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_weather_no_api_key(self):
        from src.mcp.servers.travel import get_weather
        from src.config.settings import settings

        settings.weatherapi_key = None

        result = await get_weather("Kathmandu")
        assert result.error is not None
        assert "WEATHERAPI_KEY" in result.error


class TestSearchFlights:
    @pytest.mark.asyncio
    async def test_flights_no_api_key(self):
        from src.mcp.servers.travel import search_flights
        from src.config.settings import settings

        settings.rapidapi_key = None

        result = await search_flights("KTM", "PKR", "2026-04-01")
        assert result.status == "unavailable"

    def test_flights_fallback_data(self):
        """When API fails, fallback Nepal domestic data is injected."""
        from src.mcp.servers.travel import _NEPAL_DOMESTIC_FALLBACK

        assert ("KTM", "PKR") in _NEPAL_DOMESTIC_FALLBACK
        assert ("PKR", "KTM") in _NEPAL_DOMESTIC_FALLBACK
        assert len(_NEPAL_DOMESTIC_FALLBACK[("KTM", "PKR")]) > 0


class TestSearchHotels:
    @pytest.mark.asyncio
    async def test_hotels_no_api_key(self):
        from src.mcp.servers.travel import search_hotels
        from src.config.settings import settings

        settings.rapidapi_key = None

        result = await search_hotels("Pokhara", "2026-04-01", "2026-04-03")
        assert result.status == "unavailable"
