from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from typing import Optional

class Settings(BaseSettings):

    # MCP servers
    mcp_travel_port: int = 8001
    mcp_comms_port: int = 8002
    mcp_moodboard_port: int = 8003

    # External APIs
    groq_token: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None
    
    amadeus_client_id: Optional[SecretStr] = None
    amadeus_client_secret: Optional[SecretStr] = None
    weatherapi_key: Optional[str] = None
    whatsapp_phone_number_id: Optional[str] = None
    whatsapp_access_token: Optional[SecretStr] = None
    fal_api_key: Optional[SecretStr] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

settings = Settings()
