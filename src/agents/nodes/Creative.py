import json
from langchain_core.messages import SystemMessage, AIMessage
from src.agents.state import AgentState, CreativeExtraction
from src.mcp_internals.client import get_mcp_tools
from src.config.settings import settings


# ── LLM ───────────────────────────────────────────────────────────────────
# qwen3-32b: excellent at creative prompt expansion + structured output
# Falls back to llama-3.3-70b-versatile if qwen3 fails

def get_creative_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.7,
        reasoning_effort="none",   # top-level param, NOT model_kwargs
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )

def get_creative_fallback_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=settings.groq_token.get_secret_value() if settings.groq_token else None,
    )


# ── System Prompt ─────────────────────────────────────────────────────────

CREATIVE_SYSTEM = """You are a visual prompt engineer for AI image generation.

Expand the user's creative request into a richly descriptive visual prompt
suitable for a diffusion model (fal.ai / FLUX).

Include: lighting, mood, colour palette, setting, time of day, style references,
camera language (wide shot, bokeh, cinematic, etc.)

Nepal context: if the subject involves Nepal, weave in authentic elements —
Himalayan peaks, prayer flags, terraced fields, pagodas, local markets etc.

Keep `visual_prompt` under 120 words. Set `count` to 2 if the user wants variety,
else 1.

Respond with JSON only.
"""


# ── Node ──────────────────────────────────────────────────────────────────

async def creative_agent_node(state: AgentState) -> dict:
    """Expand prompt -> generate moodboard -> respond with image URLs"""

    messages = state.get("messages", [])

    extracted: CreativeExtraction | None = None
    for name, llm_factory in [("qwen3-32b", get_creative_llm), ("llama-3.3-70b (fallback)", get_creative_fallback_llm)]:
        try:
            llm = llm_factory()
            structured_llm = llm.with_structured_output(CreativeExtraction)
            extracted = await structured_llm.ainvoke(
                [SystemMessage(content=CREATIVE_SYSTEM)] + messages
            )
            print(f"[Creative] Prompt expanded via {name}: {extracted.visual_prompt[:60]}...")
            break
        except Exception as e:
            print(f"[Creative] {name} failed: {e}, trying next...")
            continue

    if extracted is None:
        err = "Couldn't parse your creative request. Try describing the visual mood you want."
        return {"messages": [AIMessage(content=err)], "final_response": err}

    # Call moodboard tool
    tools         = await get_mcp_tools(servers=["moodboard"])
    moodboard_tool = next((t for t in tools if t.name == "generate_moodboard"), None)

    if not moodboard_tool:
        err = "⚠️ Moodboard tool not available. Is the moodboard MCP server running?"
        return {"messages": [AIMessage(content=err)], "final_response": err}

    try:
        raw    = await moodboard_tool.ainvoke({
            "prompt": extracted.visual_prompt,
            "count":  extracted.count,
        })
        result = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        err = f"Image generation failed: {e}"
        return {"messages": [AIMessage(content=err)], "final_response": err}

    if result.get("error"):
        err = f"Image generation error: {result['error']}"
        return {"messages": [AIMessage(content=err)], "final_response": err}

    images = result.get("images", [])
    if not images:
        err = "No images were generated."
        return {"messages": [AIMessage(content=err)], "final_response": err}

    # Format response
    lines = [
        "🎨 Here's your moodboard!\n",
        f"**Prompt used:** _{extracted.visual_prompt}_",
    ]
    for i, img in enumerate(images, 1):
        lines.append(f"**Image {i}:** {img['image_url']}")

    lines.append(
        "\n_Images are valid for 7 days. "
        "Let me know if you'd like a different style or mood!_"
    )

    response_text = "\n".join(lines)
    return {
        "messages": [AIMessage(content=response_text)],
        "final_response": response_text,
    }