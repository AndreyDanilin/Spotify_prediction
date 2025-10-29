# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uvicorn

# Глобальные переменные для моделей
pipeline = None
sentence_model = None

# --- Приведение decade_of_release к обученному формату ---
def normalize_decade(value):
    # если значение числовое — сокращаем до десятилетия
    try:
        val = int(value)
        if val >= 1960 and val < 1970:
            return '60'
        elif val >= 1970 and val < 1980:
            return '70'
        elif val >= 1980 and val < 1990:
            return '80'
        elif val >= 1990 and val < 2000:
            return '90'
        elif val >= 2000 and val < 2010:
            return '0'
        elif val >= 2010 and val < 2020:
            return '10'
        else:
            return 'unknown'
    except Exception:
        # если уже строка, возвращаем как есть
        return str(value)

# Загрузка моделей при старте приложения
@asynccontextmanager
async def lifespan(app):
    global pipeline, sentence_model

    try:
        # Загрузка предварительно обученного pipeline
        pipeline = joblib.load('xgb_pipe.joblib')
        print("✅ Pipeline successfully loaded")

        # Загрузка модели для эмбеддингов текста
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformer model loaded")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise RuntimeError("Model loading failed") from e

    yield  # Приложение запущено

    # Очистка ресурсов при завершении
    print("🔄 Shutting down application...")


# Инициализация приложения
app = FastAPI(
    title="Music Track Classifier API",
    description="API для классификации музыкальных треков с использованием BERT-эмбеддингов и ML-модели",
    version="1.0",
    lifespan=lifespan
)


class TrackRequest(BaseModel):
    artist: str
    decade_of_release: str
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    time_signature: int
    chorus_hit: float
    sections: int
    track_emb: List[float]

class BatchRequest(BaseModel):
    items: List[TrackRequest]


class BatchRequest(BaseModel):
    items: List[TrackRequest]


@app.post("/predict")
async def predict(request: TrackRequest):
    try:
        # Преобразуем входные данные в DataFrame
        input_data = request.dict()
        # Если track_emb упакован в список, развернем его
        if isinstance(input_data.get('track_emb'), list):
            for i, v in enumerate(input_data['track_emb']):
                input_data[f'track_emb_{i}'] = v
            del input_data['track_emb']
        input_df = pd.DataFrame([input_data])

        # --- Единственный шаг обработки: вызов pipeline ---
        prediction = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)

        return {
            "prediction": int(prediction[0]),
            "probabilities": probabilities[0].tolist(),
            "input_shape": input_df.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# --- Эндпоинт для пакетного предсказания ---
@app.post("/batch_predict")
async def batch_predict(request: BatchRequest):
    try:
        input_data = []
        for item in request.items:
            d = item.dict()
            # Если track_emb — список, развернем в отдельные столбцы
            if isinstance(d.get('track_emb'), list):
                for i, v in enumerate(d['track_emb']):
                    d[f'track_emb_{i}'] = v
                del d['track_emb']
            input_data.append(d)
        input_df = pd.DataFrame(input_data)
        predictions = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)

        results = [
            {
                "prediction": int(predictions[i]),
                "probabilities": probabilities[i].tolist()
            }
            for i in range(len(predictions))
        ]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")


# Health check эндпоинт
@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "model_loaded": hasattr(pipeline, "predict"),
        "embedding_model_loaded": hasattr(sentence_model, "encode")
    }


# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=120
    )