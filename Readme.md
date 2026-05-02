<div align="center">

# 🎵 Spotify Hit Prediction

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

### 📊 Data Source

Data is sourced from [Kaggle Dataset](https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset) and contains musical characteristics of tracks from 1960 to 2019.

## 🏗️ Project Structure

```
Spotify_prediction/
├── 📓 Spotify_prediction.ipynb    # Main notebook with analysis and training
├── 📄 Research_report.md          # Detailed research report
├── 🌐 music-classifier/           # FastAPI application
│   ├── app/                       # Main API code
│   ├── Dockerfile                 # Docker configuration
│   ├── docker-compose.yml         # Docker Compose settings
│   └── requirements.txt           # Python dependencies
├── 📁 data/                       # Source data
└── 📁 assets/                     # Images and resources
```

## 🚀 Quick Start

### 1. 📊 Data Analysis

Open the main notebook to explore the research:
```bash
python -V  # expected: Python 3.13.x
jupyter notebook Spotify_prediction.ipynb
```

### 2. 🌐 API Launch

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

API will be available at: `http://localhost:8000`

### 3. 🧪 API Testing
```bash
cd music-classifier
python test_api.py
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

## 🔧 Technology Stack

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

## 📚 Documentation

- **[📓 Main Research](Spotify_prediction.ipynb)** - complete data analysis and model training
- **[📄 Research Report](Research_report.md)** - detailed description of methodology and results
- **[🌐 API Documentation](music-classifier/README.md)** - API usage guide
- **[⚙️ Training Entry Point](pipeline_generator.py)** - canonical retraining script for `hit_ensemble.joblib`

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API and model status check |
| `/predict` | POST | Single track prediction |
| `/batch_predict` | POST | Batch prediction |

## 🤝 Contributing

This project was created for educational purposes to demonstrate data science skills and ML solution development. Suggestions for improvements are welcome!

## 📄 License

This project is distributed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

---
