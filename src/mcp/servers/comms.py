import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import Optional
from src.config.settings import settings
import httpx

mcp = FastMCP(
    "comms", host=settings.mcp_host, port=settings.mcp_comms_port, json_response=True
)


class TelegramResult(BaseModel):
    status: str
    message_id: Optional[str] = None
    error: Optional[str] = None


@mcp.tool()
async def send_telegram_message(
    body: str, chat_id: Optional[str] = None
) -> TelegramResult:
    """Send a message to the user via Telegram Bot API.

    Use this to deliver reminders, travel summaries, alerts, or any
    notification to the user's Telegram. Outbound only.

    Call this tool when:
    - The user asks to be reminded about something
    - You have built a travel summary and the user wants it sent
    - A scheduled notification needs to be delivered

    Args:
        body:    Message text to send. Markdown supported (bold, italic, code).
                 Max 4096 characters.
        chat_id: Override chat ID. If not provided, uses TELEGRAM_CHAT_ID from config.

    Returns:
        TelegramResult with status "sent" and message_id on success,
        or status "error" with an error description on failure.
    """
    target_chat_id = chat_id or settings.telegram_chat_id
    if not target_chat_id:
        return TelegramResult(
            status="error",
            error="No chat_id provided and TELEGRAM_CHAT_ID not set in .env",
        )

    if not settings.telegram_bot_token:
        return TelegramResult(
            status="error", error="TELEGRAM_BOT_TOKEN not set in .env"
        )

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token.get_secret_value()}/sendMessage"

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                url,
                json={
                    "chat_id": target_chat_id,
                    "text": body,
                    "parse_mode": "Markdown",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return TelegramResult(
                status="sent",
                message_id=str(data["result"]["message_id"]),
            )
        except httpx.HTTPStatusError as e:
            return TelegramResult(
                status="error",
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return TelegramResult(status="error", error=str(e))


if __name__ == "__main__":
    print(f"[MCP Comms] running on {settings.mcp_host}:{settings.mcp_comms_port}")
    mcp.run(transport="streamable-http")
