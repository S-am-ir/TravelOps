"""Tests for utility functions — date parsing, airport resolution, cost calculation."""

import pytest
from datetime import date
from src.agents.utils import (
    parse_natural_date,
    calculate_total_cost,
    is_within_budget,
    resolve_airport_code,
    format_flight_time,
    format_duration,
)


class TestParseNaturalDate:
    def test_iso_date_passthrough(self):
        assert parse_natural_date("2026-03-15") == "2026-03-15"

    def test_empty_returns_none(self):
        assert parse_natural_date("") is None
        assert parse_natural_date(None) is None

    def test_invalid_returns_none(self):
        assert parse_natural_date("not a date at all xyz") is None


class TestResolveAirportCode:
    def test_known_city(self):
        assert resolve_airport_code("Kathmandu") == "KTM"
        assert resolve_airport_code("pokhara") == "PKR"

    def test_iata_passthrough(self):
        assert resolve_airport_code("KTM") == "KTM"
        assert resolve_airport_code("PKR") == "PKR"

    def test_unknown_city_returns_none(self):
        assert resolve_airport_code("New York") is None

    def test_empty_returns_none(self):
        assert resolve_airport_code("") is None
        assert resolve_airport_code(None) is None


class TestCostCalculation:
    def test_basic_cost(self):
        result = calculate_total_cost(5000, 2000, 3)
        assert result["flight"] == 5000
        assert result["hotel"] == 6000
        assert result["total"] == 11000

    def test_zero_nights(self):
        result = calculate_total_cost(5000, 2000, 0)
        assert result["hotel"] == 0
        assert result["total"] == 5000


class TestBudgetCheck:
    def test_within_budget(self):
        assert is_within_budget(9000, 10000)

    def test_over_budget(self):
        assert not is_within_budget(11000, 10000)

    def test_zero_budget(self):
        assert not is_within_budget(100, 0)


class TestFormatHelpers:
    def test_flight_time(self):
        result = format_flight_time("2026-03-15T07:30:00")
        assert "07:30" in result

    def test_duration(self):
        assert format_duration(195) == "3h 15m"
        assert format_duration(120) == "2h"
        assert format_duration(45) == "0h 45m"
