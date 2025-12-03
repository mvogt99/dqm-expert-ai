"""
Configuration with environment variable support and validation.
"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Database
    database_url: str = "postgresql+asyncpg://dqm_user:dqm_password@localhost:5432/northwind"
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Local AI endpoints
    local_ai_planning_url: str = "http://localhost:8004/v1"  # RTX 5090
    local_ai_coding_url: str = "http://localhost:8015/v1"    # RTX 3050
    
    # App settings
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
