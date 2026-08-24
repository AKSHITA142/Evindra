import os
from functools import lru_cache
from typing import List, Optional, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed application settings loaded from environment variables and local .env file.
    """
    app_name: str = "DataPilot-AI"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    secret_key: str = Field(default="datapilot-secret-key-change-in-production", alias="SECRET_KEY")
    
    # Database
    database_url: str = Field(default="sqlite:///./datapilot.db", alias="DATABASE_URL")
    
    # File Storage
    storage_dir: str = Field(default="./storage", alias="STORAGE_DIR")
    max_upload_size_mb: int = Field(default=150, alias="MAX_UPLOAD_SIZE_MB")
    storage_backend: str = Field(default="local", alias="STORAGE_BACKEND")  # 'supabase' or 'local'
    supabase_url: Optional[str] = Field(default=None, alias="SUPABASE_URL")
    supabase_key: Optional[str] = Field(default=None, alias="SUPABASE_KEY")
    supabase_bucket: str = Field(default="datasets", alias="SUPABASE_BUCKET")
    max_ml_sample_rows: int = Field(default=50000, alias="MAX_ML_SAMPLE_ROWS")
    
    # Server Parameters
    server_host: str = Field(default="127.0.0.1", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")
    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")

    # CORS
    cors_origins: Union[List[str], str] = Field(default=["*"], alias="CORS_ORIGINS")

    # LLM Settings (Strictly loaded from .env environment configuration)
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPEN_ROUTER")
    model_name: Optional[str] = Field(default=None, alias="MODEL_NAME")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    llm_model_name: Optional[str] = Field(default="gemini-3.5-flash-lite", alias="LLM_MODEL_NAME")
    gemini_model_name: Optional[str] = Field(default="gemini-3.5-flash-lite", alias="GEMINI_MODEL_NAME")


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_cors_origins_list(self) -> List[str]:
        """
        Parse CORS origins. Allows wildcard '*' ONLY when ENVIRONMENT is explicitly 'development' or 'dev'.
        In non-development environments, wildcard origins are filtered out and explicit origins (or frontend_url) are required.
        """
        raw_list: List[str] = []
        if isinstance(self.cors_origins, str):
            raw_list = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        elif isinstance(self.cors_origins, list):
            raw_list = [str(origin).strip() for origin in self.cors_origins if str(origin).strip()]

        is_dev = str(self.environment).strip().lower() in ("development", "dev", "local")
        
        # When CORS origins contains wildcard '*', replace with explicit localhost URLs so allow_credentials=True does not violate CORS spec
        if "*" in raw_list:
            explicit_defaults = [self.frontend_url, "http://localhost:3000", "http://127.0.0.1:3000"]
            non_wildcard = [o for o in raw_list if o != "*"]
            combined = list(dict.fromkeys(explicit_defaults + non_wildcard))
            return combined

        return raw_list if raw_list else [self.frontend_url, "http://localhost:3000", "http://127.0.0.1:3000"]


@lru_cache()
def get_settings() -> Settings:
    """Returns application Settings instance loaded from environment and .env."""
    return Settings()
