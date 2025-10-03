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

# Загрузка моделей при старте приложения
@asynccontextmanager
async def lifespan():
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


# Модель для валидации входных данных
class TrackRequest(BaseModel):
    artist: str
    track: str
    decade_of_release: int
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


class BatchRequest(BaseModel):
    items: List[TrackRequest]


# Эндпоинт для одиночного предсказания
@app.post("/predict")
async def predict(request: TrackRequest):
    try:
        # Конвертация в DataFrame
        input_df = pd.DataFrame([request.dict()])

        # Генерация эмбеддингов для трека
        track_embedding = sentence_model.encode(
            input_df['track'].values,
            show_progress_bar=False
        )

        # Добавление эмбеддингов в данные
        for i in range(track_embedding.shape[1]):
            input_df[f'track_embed_{i}'] = track_embedding[:, i]

        # Удаление исходного текстового поля
        input_df = input_df.drop(columns=['track'])

        # Предсказание
        prediction = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)

        return {
            "prediction": int(prediction[0]),
            "probabilities": probabilities[0].tolist(),
            "track_embedding_dim": track_embedding.shape[1]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# Эндпоинт для пакетной обработки
@app.post("/batch_predict")
async def batch_predict(request: BatchRequest):
    try:
        # Конвертация в DataFrame
        input_data = [item.dict() for item in request.items]
        input_df = pd.DataFrame(input_data)

        # Генерация эмбеддингов для всех треков
        track_embeddings = sentence_model.encode(
            input_df['track'].values,
            show_progress_bar=False,
            batch_size=32
        )

        # Добавление эмбеддингов в данные
        for i in range(track_embeddings.shape[1]):
            input_df[f'track_embed_{i}'] = track_embeddings[:, i]

        # Удаление исходного текстового поля
        input_df = input_df.drop(columns=['track'])

        # Пакетное предсказание
        predictions = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)

        # Формирование результата
        results = []
        for i in range(len(predictions)):
            results.append({
                "prediction": int(predictions[i]),
                "probabilities": probabilities[i].tolist(),
                "track": request.items[i].track  # Возвращаем исходное название
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


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