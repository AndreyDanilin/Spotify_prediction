from pathlib import Path

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "hit_ensemble.joblib"

@dataclass(frozen=True)
class Settings:
    MODEL_VERSION = "2.0"
    API_PREFIX = "/api"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

settings = Settings()
