import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv('LLM', 'llama-3.1-8b-instant')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DOC_DIRECTORY = Path(os.getenv('DIRECTORY', r"G:\datato_ingestion"))
DB_CONNECTION = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg://postgres:mysecretpassword@localhost:5432/vectordb"
)
COLLECTION_NAME = "rag_agent_documents"


"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.
"""
import os
from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load raw .env into environment variables first
load_dotenv()

class Settings(BaseSettings):
    # LLM Configuration
    groq_api_key: str=os.getenv("GROQ_API_KEY", "")
    groq_api_fallback: str=os.getenv("GROQ_API_FALLBACK", "") 
    primary_model: str = os.getenv("LLM", "qwen/qwen3.8-27b")
    fallback_model: str = os.getenv("fallback_llm", "deepseek-chat")
    
    
    # LangSmith / LangChain Tracing
    langchain_tracing_v2: bool = True
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")     
    langchain_project: str = "advance_rag"
    
    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance that populates os.environ for SDK compatibility."""
    settings = Settings()
    
    # Export fields explicitly to os.environ so LangChain/LangSmith SDKs pick them up
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["GROQ_API_KEY"] = settings.groq_api_key
    os.environ["GROQ_API_FALLBACK"] = settings.groq_api_fallback
    
    return settings