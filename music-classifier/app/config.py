from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("model.joblib")))
    fallback_model_path: Path = Path(__file__).with_name("xgb_pipe.joblib")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    model_version: str = os.getenv("MODEL_VERSION", "local")


settings = Settings()
