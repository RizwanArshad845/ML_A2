"""All settings come from environment variables (Task 4.4), with safe defaults for local use.

Nothing secret or machine-specific is written in code: the same image runs locally,
in Docker Compose and in the cloud just by changing the environment.
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _optional_float(name: str):
    value = os.getenv(name, "").strip()
    return float(value) if value else None


@dataclass
class Settings:
    model_path: str
    threshold: float | None  # None = use threshold.json next to the model, else 0.5
    database_url: str
    max_upload_mb: float
    model_name: str
    model_version: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        default_db = (ROOT / "instance" / "visionguard.db").as_posix()
        return cls(
            model_path=os.getenv("MODEL_PATH", str(ROOT / "models" / "champion" / "model.keras")),
            threshold=_optional_float("THRESHOLD"),
            database_url=os.getenv("DATABASE_URL", f"sqlite:///{default_db}"),
            max_upload_mb=float(os.getenv("MAX_UPLOAD_MB", "5")),
            model_name=os.getenv("MODEL_NAME", "visionguard"),
            model_version=os.getenv("MODEL_VERSION", "local"),
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "5000")),
        )
