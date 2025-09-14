from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Math Routing Agent"
    
    # Database Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "math_agent"
    
    # Vector Database Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # LLM Settings
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    # MCP Settings
    TAVILY_API_KEY: str = ""
    
    # Security
    SECRET_KEY: str = "your-secret-key"
    
    # Guardrails
    ENABLE_GUARDRAILS: bool = True
    MAX_TOKENS_PER_MINUTE: int = 10000
    
    class Config:
        env_file = ".env"

settings = Settings()