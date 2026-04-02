from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from typing import Optional


class Settings(BaseSettings):
    # MCP server ports
    mcp_travel_port: int = 8001
    mcp_comms_port: int = 8002
    mcp_search_port: int = 8004

    # MCP host
    mcp_host: str = "127.0.0.1"
    mcp_travel_host: Optional[str] = None
    mcp_comms_host: Optional[str] = None
    mcp_search_host: Optional[str] = None

    # Postgres checkpointer (Supabase connection string)
    supabase_url: Optional[SecretStr] = None

    # LLM providers
    groq_token: Optional[SecretStr] = None
    openrouter_api_key: Optional[SecretStr] = None

    # Travel APIs (RapidAPI key covers Sky-Scrapper + Booking.com15)
    rapidapi_key: Optional[SecretStr] = None
    weatherapi_key: Optional[SecretStr] = None

    # Web search
    tavily_api_key: Optional[SecretStr] = None

    # Image generation
    fal_api_key: Optional[SecretStr] = None

    # Auth
    jwt_secret: Optional[SecretStr] = None

    @property
    def mcp_travel_resolved_host(self) -> str:
        return self.mcp_travel_host or self.mcp_host

    @property
    def mcp_comms_resolved_host(self) -> str:
        return self.mcp_comms_host or self.mcp_host

    @property
    def mcp_search_resolved_host(self) -> str:
        return self.mcp_search_host or self.mcp_host

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
