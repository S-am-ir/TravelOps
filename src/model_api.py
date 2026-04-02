"""
model_api.py — Shared LLM factory for all agent nodes.

Fallback chain:
  1. Groq: openai/gpt-oss-120b (primary — fast, paid, reliable)
  2. Groq: qwen/qwen3-32b (fallback — fast, paid)
  3. OpenRouter: qwen/qwen3.6-plus-preview:free (last resort — retries on rate limit)
"""

import asyncio
from src.config.settings import settings

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _groq_llm(model: str, temperature: float = 0):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )


def _openrouter_llm(model: str, temperature: float = 0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=OPENROUTER_BASE,
        api_key=settings.openrouter_api_key.get_secret_value()
        if settings.openrouter_api_key
        else None,
    )


def get_primary_llm(temperature: float = 0):
    """Groq openai/gpt-oss-120b — fast, reliable, no retries needed."""
    return _groq_llm("openai/gpt-oss-120b", temperature)


def get_fallback1_llm(temperature: float = 0):
    """Groq qwen/qwen3-32b — fast, reliable, no retries needed."""
    return _groq_llm("qwen/qwen3-32b", temperature)


def get_fallback2_llm(temperature: float = 0):
    """OpenRouter qwen/qwen3.6-plus-preview:free — last resort with retries."""
    return _openrouter_llm("qwen/qwen3.6-plus-preview:free", temperature)


def get_fallback3_llm(temperature: float = 0):
    return get_primary_llm(temperature)


def get_fallback4_llm(temperature: float = 0):
    return get_fallback1_llm(temperature)


def get_fallback5_llm(temperature: float = 0):
    return get_fallback2_llm(temperature)


async def invoke_with_fallback(messages: list, structured_schema=None) -> tuple:
    """Try each LLM in order. Only retries on the last fallback."""

    # Groq primary — no retries
    for name, factory in [
        ("groq-gpt-oss-120b", get_primary_llm),
        ("groq-qwen3-32b", get_fallback1_llm),
    ]:
        try:
            llm = factory()
            if structured_schema:
                llm = llm.with_structured_output(structured_schema)
            return await llm.ainvoke(messages), name
        except Exception as e:
            print(f"[model_api] {name} failed: {str(e)[:100]}")
            continue

    # OpenRouter last resort — with retries
    for attempt in range(3):
        try:
            llm = get_fallback2_llm()
            if structured_schema:
                llm = llm.with_structured_output(structured_schema)
            return await llm.ainvoke(messages), "openrouter-qwen3.6-plus"
        except Exception as e:
            err = str(e)
            is_rate_limit = any(
                k in err.lower() for k in ["429", "rate", "limit", "quota"]
            )
            if is_rate_limit and attempt < 2:
                wait = 1 ** (attempt + 1)  # 1s, 1s — fast retries
                print(
                    f"[model_api] openrouter rate-limited, retry {attempt + 1}/3 in {wait}s"
                )
                await asyncio.sleep(wait)
            else:
                print(f"[model_api] openrouter failed: {str(e)[:100]}")
                break

    raise RuntimeError("All 3 LLMs failed")
