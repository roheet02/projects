"""
JobPilot AI - Application Settings
Pydantic-based settings management with .env support.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, List
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "JobPilot AI"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Database
    database_url: str = f"sqlite+aiosqlite:///{BASE_DIR}/jobpilot.db"

    # ChromaDB
    chroma_persist_path: str = str(BASE_DIR / "chroma_db")

    # Job Search
    default_job_portals: str = "linkedin,indeed,naukri"
    max_jobs_per_search: int = 50
    job_match_threshold: float = 0.65

    # Outreach
    outreach_review_required: bool = True
    max_daily_outreach: int = 20

    # Browser
    browser_headless: bool = False
    browser_slow_mo: int = 50
    playwright_timeout: int = 30000

    @property
    def job_portal_list(self) -> List[str]:
        return [p.strip() for p in self.default_job_portals.split(",")]


# Singleton
settings = Settings()
