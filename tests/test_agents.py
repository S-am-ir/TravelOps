"""Tests for agent state and intent classification."""

import pytest
from src.agents.state import (
    IntentClassification,
    ReminderExtraction,
    CreativeExtraction,
)


class TestIntentClassification:
    def test_valid_travel_intent(self):
        ic = IntentClassification(
            intent="travel_planning",
            confidence=0.95,
            reasoning="User is asking about flights",
        )
        assert ic.intent == "travel_planning"
        assert ic.confidence == 0.95

    def test_valid_reminder_intent(self):
        ic = IntentClassification(
            intent="reminder",
            confidence=0.8,
            reasoning="User wants to be reminded",
        )
        assert ic.intent == "reminder"

    def test_invalid_intent_rejected(self):
        with pytest.raises(Exception):
            IntentClassification(
                intent="invalid_intent",
                confidence=0.9,
                reasoning="test",
            )

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            IntentClassification(
                intent="travel_planning",
                confidence=1.5,
                reasoning="test",
            )

        with pytest.raises(Exception):
            IntentClassification(
                intent="travel_planning",
                confidence=-0.1,
                reasoning="test",
            )


class TestReminderExtraction:
    def test_valid_extraction(self):
        r = ReminderExtraction(
            reminder_message="Call the bank",
            scheduled_for="2026-04-01T10:00:00",
        )
        assert r.reminder_message == "Call the bank"
        assert r.scheduled_for == "2026-04-01T10:00:00"
        assert r.repeat_rule == "none"

    def test_now_scheduled(self):
        r = ReminderExtraction(
            reminder_message="Test",
            scheduled_for="now",
        )
        assert r.scheduled_for == "now"

    def test_daily_repeat(self):
        r = ReminderExtraction(
            reminder_message="Daily standup",
            scheduled_for="2026-04-01T09:00:00",
            repeat_rule="daily",
        )
        assert r.repeat_rule == "daily"


class TestCreativeExtraction:
    def test_valid_extraction(self):
        c = CreativeExtraction(
            visual_prompt="A beautiful sunset over the Himalayas",
            count=2,
        )
        assert c.count == 2
        assert "Himalayas" in c.visual_prompt

    def test_default_count(self):
        c = CreativeExtraction(
            visual_prompt="A serene lake",
        )
        assert c.count == 1

    def test_count_bounds(self):
        with pytest.raises(Exception):
            CreativeExtraction(
                visual_prompt="test",
                count=5,
            )
