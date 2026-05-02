# Music Classifier API

An API for classifying music tracks using BERT embeddings and a weighted ensemble model.

## Description

This project is a web API for predicting music hit probability from audio features and track names. The API computes track-name embeddings internally and uses a pre-trained weighted soft-voting ensemble.

The supported runtime target is **Python 3.13**.

## Features

- 🎵 Single prediction for one track
- 📦 Batch processing for multiple tracks
- 🔍 Health check for API status monitoring
- 🐳 Docker containerization for easy deployment

## Project Structure

```
music-classifier/
├── app/
│ ├── main.py # Main FastAPI application
│ ├── config.py # Application configuration
│ └── hit_ensemble.joblib # Pre-trained weighted ensemble
├── scripts/
│ └── start.sh # Server startup script
├── test_api.py # Test script
├── requirements.txt # Python dependencies
├── Dockerfile # Docker image
├── docker-compose.yml # Docker Compose configuration
└── README.md # This file
```

## Quick Start

The API expects `app/hit_ensemble.joblib`. Build it from the repository root before starting the server:

```bash
python pipeline_generator.py \
  --hf-home .hf-cache \
  --offline-embeddings \
  --tabm-device cuda \
  --tree-device cuda
```

Omit `--offline-embeddings` on the first run if the `all-MiniLM-L6-v2` embedding model is not cached yet.

### Local Launch

1. **Install dependencies:**
   ```bash
   python -V  # expected: Python 3.13.x
   cd music-classifier
   pip install -r requirements.txt
   ```

2. **Start the application:**
   ```bash
   python app/main.py
   ```

3. **Test the functionality:**
   ```bash
   python test_api.py
   ```

### Docker Launch

1. **Build and run with Docker Compose:**
   ```bash
   cd music-classifier
   docker-compose up --build
   ```

2. **Test the functionality:**
   ```bash
   python test_api.py
   ```

## API Endpoints

### Health Check
```
GET /health
```
Test of python app/main.py and loaded models.

**Response:**
```json
{
  "status": "OK",
  "model_loaded": true,
  "embedding_model_loaded": true,
  "model_version": "2.0"
}
```

### Single Prediction
```
POST /predict
```

**Request body:**
```json
{
  "artist": "The Beatles",
  "track": "Hey Jude",
  "decade_of_release": 1960,
  "danceability": 0.5,
  "energy": 0.7,
  "key": 7,
  "loudness": -8.5,
  "mode": 1,
  "speechiness": 0.03,
  "acousticness": 0.2,
  "instrumentalness": 0.0,
  "liveness": 0.1,
  "valence": 0.8,
  "tempo": 120.0,
  "duration_ms": 431000,
  "time_signature": 4,
  "chorus_hit": 0.5,
  "sections": 8
}
```

**Response:**
```json
{
  "prediction": 1,
  "probabilities": [0.2, 0.8],
  "model_version": "2.0",
  "track_embedding_dim": 384
}
```

### Batch Prediction
```
POST /batch_predict
```

**Request body:**
```json
{
  "items": [
    {
      "artist": "Queen",
      "track": "Bohemian Rhapsody",
      "decade_of_release": 1970,
      "danceability": 0.3,
      "energy": 0.6,
      "key": 0,
      "loudness": -7.2,
      "mode": 1,
      "speechiness": 0.05,
      "acousticness": 0.1,
      "instrumentalness": 0.0,
      "liveness": 0.2,
      "valence": 0.4,
      "tempo": 72.0,
      "duration_ms": 355000,
      "time_signature": 4,
      "chorus_hit": 0.3,
      "sections": 12
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "prediction": 0,
      "probabilities": [0.7, 0.3],
      "track": "Bohemian Rhapsody",
      "model_version": "2.0"
    }
  ]
}
```

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `artist` | string | Artist name |
| `track` | string | Track title |
| `decade_of_release` | integer | Release decade (1960, 1970, etc.) |
| `danceability` | float | Danceability (0.0-1.0) |
| `energy` | float | Energy level (0.0-1.0) |
| `key` | integer | Musical key (0-11) |
| `loudness` | float | Loudness in dB |
| `mode` | integer | Mode (0=minor, 1=major) |
| `speechiness` | float | Accepted for compatibility, ignored by the trained model |
| `acousticness` | float | Acousticness (0.0-1.0) |
| `instrumentalness` | float | Accepted for compatibility, ignored by the trained model |
| `liveness` | float | Liveness (0.0-1.0) |
| `valence` | float | Positivity (0.0-1.0) |
| `tempo` | float | Tempo in BPM |
| `duration_ms` | integer | Duration in milliseconds |
| `time_signature` | integer | Time signature (3, 4, 5, etc.) |
| `chorus_hit` | float | Chorus hit probability |
| `sections` | integer | Number of sections |

## Testing

Run the test script to verify all endpoints:

```bash
python test_api.py
```

## Monitoring

The API includes a health check endpoint for status monitoring:
- ML model loading verification
- Embedding model loading verification
- Model version
- Overall application status

## Technologies

- **FastAPI** - Web framework for API creation
- **Weighted ensemble** - Soft voting over the best validation models
- **XGBoost / CatBoost / Logistic Regression / TabM** - Candidate model families
- **Sentence Transformers** - BERT embeddings for text
- **Python 3.13** - target runtime for local and containerized deployment
- **Pandas** - Data processing
- **Docker** - Containerization
- **Uvicorn** - ASGI server

## License

This project is part of a data analysis research project.


