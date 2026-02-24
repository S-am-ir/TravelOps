import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import List
from src.config.settings import settings
import httpx

mcp = FastMCP("moodboard", json_response=True)

class MoodboardImage(BaseModel):
    prompt_used: str
    image_url: str
    seed: int

class MoodboardResult(BaseModel):
    images: List[MoodboardImage]
    error: str | None = None

@mcp.tool()
async def generate_moodboard(prompt: str, count: int = 1) -> MoodboardResult:
    """Generate a moodboard of AI images based on a descriptive prompt.

    For example, if the user wants a romantic surprise for their girlfriend,
    the prompt passed here should already describe the visual mood, setting,
    lighting, and aesthetic — not just "romantic surprise".

    Args:
        prompt: A rich, descriptive visual prompt for the moodboard theme.
                E.g. "soft candlelight dinner by a lakeside at golden hour,
                rose petals, warm bokeh, cinematic and intimate mood"
        count:  Number of images to generate (default 1, max 2).
                Each image uses the same prompt with different seeds
                for visual variety.

    Returns:
        MoodboardResult with a list of MoodboardImage entries, each containing
        the image URL (valid for 7 days), the prompt used, and the seed.
        Returns error field populated on failure.

    Example:
        generate_moodboard(
            prompt="misty mountain trek at dawn, prayer flags, golden light through clouds, Nepal Himalayas, cinematic wide shot",
            count=2
        )
    """
    count = max(1, min(count, 2))
    url = f"https://fal.run/fal-ai/flux/schnell"
    headers = {
        "Authorization": f"Key {settings.fal_api_key.get_secret_value()}",
        "Content-Type": "application/json"
    }

    images = []
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(count):
            try:
                resp = await client.post(url, headers=headers, json={
                    "prompt": prompt,
                    "image_size": "landscape_4_3",
                    "num_inference_steps": 4,
                    "num_images": 1,
                    "enable_safety_checker": True,
                })

                if resp.status_code != 200:
                    return MoodboardResult(
                        images=images,
                        error=f"fal.ai error {resp.status_code}: {resp.text}",
                    )
                
                data = resp.json()
                img = data["images"][0]
                images.append(MoodboardImage(
                    prompt_used=prompt,
                    image_url=img["url"],
                    seed=data.get("seed", i),
                ))

            except Exception as e:
                return MoodboardResult(images=images, error=str(e))
            
if __name__ == "__main__":
    print(f"[MCP Moodboard] running on port {settings.mcp_moodboard_port}")
    mcp.run(transport="streamable-http")