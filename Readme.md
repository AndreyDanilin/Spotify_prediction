<div align="center">

# Spotify Hit Prediction

**Predicting Spotify hits using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange.svg)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

<img src="assets/spotify_logo.png" alt="Spotify Logo" width="200"/>

</div>

## 📋 About the Project

This project is a comprehensive solution for predicting musical hits based on Spotify data. The project includes data exploration, training of various machine learning models, and deployment of a web API for practical use.

The project now targets **Python 3.13** for notebook work, training, and API deployment.

### 🎯 Main Objectives

- **Data Analysis**: Research on musical characteristics that influence track popularity
- **Modeling**: Comparison of various machine learning algorithms to select the optimal one
- **Production**: Creation of a ready-to-use API backed by a trained weighted ensemble

## Project Structure

```text
Spotify_prediction/
├── src/spotify_prediction/       # Package: features, training, model service, Litestar API
├── music-classifier/             # Docker/API wrapper and smoke test
├── data/                         # Kaggle Spotify hit predictor CSV files
├── Spotify_prediction.ipynb      # Research notebook with v2 ensemble section
├── pipeline_generator.py         # Compatibility training entrypoint
├── pyproject.toml                # Python 3.12 dependency groups
└── tests/                        # Lightweight unit/API tests
```

## Local Development

```bash
python -V  # expected: Python 3.13.x
jupyter notebook Spotify_prediction.ipynb
```

The `-s` flag avoids a pytest capture issue on some WSL-mounted Windows paths.

Generate the ensemble artifact before starting the API:
```bash
python pipeline_generator.py \
  --hf-home .hf-cache \
  --tabm-device cuda \
  --tree-device cuda
```

For a repeat run after the Hugging Face embedding model is already cached:
```bash
python pipeline_generator.py \
  --hf-home .hf-cache \
  --offline-embeddings \
  --tabm-device cuda \
  --tree-device cuda
```

The script writes `music-classifier/app/hit_ensemble.joblib` and caches generated training features at `artifacts/training_features.joblib`.

#### Local Launch
```bash
python -V  # expected: Python 3.13.x
cd music-classifier
pip install -r requirements.txt
python app/main.py
```

#### Docker Launch
```bash
cd music-classifier
docker-compose up --build
```

Full training uses the heavier ML stack:

```bash
uv run --extra train spotify-train --data-dir data --output music-classifier/app/model.joblib
```

## 🔬 Methodology

### Machine Learning Algorithms
- **Weighted Soft-Voting Ensemble** (main model) - combines the strongest validation models by ROC-AUC
- **XGBoost** - gradient boosting
- **Random Forest** - ensemble of decision trees
- **Logistic Regression** - logistic regression
- **TabM** - parameter-efficient neural ensemble for tabular data
- **SVM** - support vector machines
- **CatBoost** - gradient boosting by Yandex

### Data Processing
- **BERT embeddings** for track names
- **Feature cache** for reusable track embeddings and generated model features
- **Feature engineering** for musical characteristics
- **Cross-validation** for model quality assessment
- **Hyperparameter optimization** with Optuna
- **Weighted soft voting** using validation ROC-AUC for model selection and weights

## 📈 Results

- **Best model**: weighted ensemble selected by ROC-AUC
- **Important features**: danceability, energy, valence, tempo
- **Time coverage**: 60 years of musical history (1960-2019)

## Run The API

### Data Science
- **Python** - main programming language
- **Pandas** - data processing
- **NumPy** - numerical computations
- **Scikit-learn** - machine learning
- **XGBoost** - gradient boosting
- **CatBoost** - gradient boosting
- **TabM / PyTorch** - tabular neural modeling
- **Polars** - optional fast CSV loading before pandas/numpy/torch conversion

### API and Deployment
- **FastAPI** - modern web framework
- **Docker** - containerization
- **Uvicorn** - ASGI server
- **Pydantic** - data validation

### Visualization
- **Matplotlib** - basic plots
- **Seaborn** - statistical visualization
- **Plotly** - interactive plots

Endpoints:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Model and embedding status |
| POST | `/predict` | Single track prediction |
| POST | `/batch_predict` | Batch prediction |

Example request:

```json
{
  "track": "Hey Jude",
  "artist": "The Beatles",
  "decade_of_release": 1968,
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

- **[📓 Main Research](Spotify_prediction.ipynb)** - complete data analysis and model training
- **[📄 Research Report](Research_report.md)** - detailed description of methodology and results
- **[🌐 API Documentation](music-classifier/README.md)** - API usage guide
- **[⚙️ Training Entry Point](pipeline_generator.py)** - canonical retraining script for `hit_ensemble.joblib`

Serving dependencies are intentionally kept inside the image:

```bash
cd music-classifier
docker compose up --build app
```

To retrain in a container profile:

```bash
cd music-classifier
docker compose --profile train run --build trainer
```

This split keeps local and CI feedback light while still giving a reproducible heavy ML environment when needed.

## Research

`Spotify_prediction.ipynb` keeps the historical analysis and adds a final v2 Polars-first ensemble section. The production trainer and notebook now share the same package functions, so research and serving do not drift as easily.
