import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    supabase_url: str
    supabase_key: str
    orchestrator_agent_url: str = "http://localhost:10010"
    chat_api_host: str = "0.0.0.0"
    chat_api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    missing = [
        name
        for name, value in {
            "SUPABASE_URL": supabase_url,
            "SUPABASE_KEY": supabase_key,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        orchestrator_agent_url=os.getenv("ORCHESTRATOR_AGENT_URL", "http://localhost:10010"),
        chat_api_host=os.getenv("CHAT_API_HOST", "0.0.0.0"),
        chat_api_port=int(os.getenv("CHAT_API_PORT", "8000")),
    )
