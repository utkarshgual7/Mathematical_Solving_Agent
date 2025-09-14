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
    
    # Vector Database Settings (Legacy - now using Agno LanceDB)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # Agno Settings
    AGNO_EMBEDDINGS_PATH: str = "knowledge_base/embeddings"
    AGNO_VECTOR_DB_TYPE: str = "lancedb"
    AGNO_SEARCH_TYPE: str = "hybrid"
    AGNO_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # LLM Settings
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    # MCP and Web Search Settings
    TAVILY_API_KEY: str = ""
    ENABLE_AGNO_SEARCH: bool = True
    ENABLE_TAVILY_FALLBACK: bool = True
    MAX_SEARCH_RESULTS: int = 5
    
    # Security
    SECRET_KEY: str = "your-secret-key"
    
    # Guardrails
    ENABLE_GUARDRAILS: bool = True
    MAX_TOKENS_PER_MINUTE: int = 10000
    
    # Knowledge Base Settings
    PDF_UPLOAD_PATH: str = "knowledge_base/pdfs"
    PDF_PROCESSED_PATH: str = "knowledge_base/processed"
    KNOWLEDGE_BASE_LOGS_PATH: str = "knowledge_base/logs"
    AUTO_PROCESS_PDFS: bool = True
    
    # Mathematical Processing Settings
    ENABLE_LATEX_PARSING: bool = True
    ENABLE_SYMPY_VALIDATION: bool = True
    MIN_PROBLEM_CONFIDENCE: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings()