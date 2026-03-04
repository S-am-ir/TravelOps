from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from typing import Optional


class Settings(BaseSettings):

    # MCP server ports
    mcp_travel_port:   int = 8001
    mcp_comms_port:    int = 8002
    mcp_moodboard_port: int = 8003

    # Postgres checkpointer (Supabase connection string)
    supabase_url: Optional[SecretStr] = None

    # LLM
    groq_token: Optional[SecretStr] = None

    # Travel APIs (RapidAPI key covers Sky-Scrapper + Booking.com15)
    rapidapi_key:   Optional[SecretStr] = None
    weatherapi_key: Optional[SecretStr] = None

    # Notifications
    telegram_bot_token: Optional[SecretStr] = None
    telegram_chat_id:   Optional[str] = None

    # Image generation
    fal_api_key: Optional[SecretStr] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()