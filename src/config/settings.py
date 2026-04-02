from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from typing import Optional


class Settings(BaseSettings):


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

    # Auth
    jwt_secret: Optional[SecretStr] = None



    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
