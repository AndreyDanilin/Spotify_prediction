# config.py
import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "xgboost_pipeline.pkl"

class Settings:
    MODEL_VERSION = "1.0"
    API_PREFIX = "/api"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

settings = Settings()