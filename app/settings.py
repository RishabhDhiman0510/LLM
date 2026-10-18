import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.getenv("WIKIGPT_MODEL_DIR", ROOT / "models" / "wikigpt-12m"))
    db_path: Path = Path(os.getenv("WIKIGPT_DB", ROOT / "app" / "data" / "chats.db"))
    device: str = os.getenv("WIKIGPT_DEVICE", "auto")
    host: str = os.getenv("WIKIGPT_HOST", "127.0.0.1")
    port: int = int(os.getenv("WIKIGPT_PORT", "8000"))
    max_new_tokens: int = int(os.getenv("WIKIGPT_MAX_NEW_TOKENS", "200"))
    temperature: float = float(os.getenv("WIKIGPT_TEMPERATURE", "0.8"))
    top_k: int = int(os.getenv("WIKIGPT_TOP_K", "40"))
    repetition_penalty: float = float(os.getenv("WIKIGPT_REPETITION_PENALTY", "1.1"))


settings = Settings()
