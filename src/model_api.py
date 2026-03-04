"""
model_api.py — Shared LLM factory for all agent nodes.

Primary:  qwen/qwen3-32b  (Groq) — reasoning, tool use, structured output
Fallback: llama-3.3-70b-versatile (Groq) — text-only, never bind tools
"""

from functools import lru_cache
from src.config.settings import settings


def _groq_key() -> str | None:
    return settings.groq_token.get_secret_value() if settings.groq_token else None


def get_primary_llm(temperature: float = 0):
    """qwen3-32b: strong structured output + tool calling. Non-thinking mode for speed."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=temperature,
        reasoning_effort="none",
        api_key=_groq_key(),
    )


def get_fallback_llm(temperature: float = 0):
    """llama-3.3-70b: text-only fallback. Never bind tools — produces broken XML calls."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=_groq_key(),
    )


async def invoke_with_fallback(messages: list, structured_schema=None) -> tuple:
    """
    Try primary LLM, fall back to llama on any error.

    Args:
        messages:          Full message list to pass to the LLM.
        structured_schema: Pydantic model for .with_structured_output(), or None for raw.

    Returns:
        (result, model_name_used)
    """
    attempts = [
        ("qwen3-32b",           lambda: get_primary_llm()),
        ("llama-3.3-70b (fallback)", lambda: get_fallback_llm()),
    ]

    last_error = None
    for name, factory in attempts:
        try:
            llm = factory()
            if structured_schema:
                llm = llm.with_structured_output(structured_schema)
            result = await llm.ainvoke(messages)
            return result, name
        except Exception as e:
            last_error = e
            print(f"[model_api] {name} failed: {e}, trying next...")
            continue

    raise RuntimeError(f"All LLMs failed. Last error: {last_error}")