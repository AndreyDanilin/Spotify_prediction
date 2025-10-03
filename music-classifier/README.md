# Music Classifier API

API для классификации музыкальных треков с использованием BERT-эмбеддингов и XGBoost модели.

## Описание

Этот проект представляет собой веб-API для предсказания музыкальных характеристик на основе аудио-фич и названий треков. API использует предобученную модель XGBoost и BERT-эмбеддинги для обработки текстовых данных.

## Возможности

- 🎵 Одиночное предсказание для одного трека
- 📦 Пакетная обработка множественных треков
- 🔍 Health check для мониторинга состояния API
- 🐳 Docker контейнеризация для простого развертывания

## Структура проекта

```
music-classifier/
├── app/
│   ├── main.py          # Основное приложение FastAPI
│   ├── config.py        # Конфигурация приложения
│   └── xgb_pipe.joblib  # Предобученная модель
├── scripts/
│   └── start.sh         # Скрипт запуска сервера
├── test_api.py          # Тестовый скрипт
├── requirements.txt     # Python зависимости
├── Dockerfile          # Docker образ
├── docker-compose.yml  # Docker Compose конфигурация
└── README.md           # Этот файл
```

## Быстрый старт

### Локальный запуск

1. **Установка зависимостей:**
   ```bash
   cd music-classifier
   pip install -r requirements.txt
   ```

2. **Запуск приложения:**
   ```bash
   python app/main.py
   ```

3. **Проверка работы:**
   ```bash
   python test_api.py
   ```

### Docker запуск

1. **Сборка и запуск с Docker Compose:**
   ```bash
   cd music-classifier
   docker-compose up --build
   ```

2. **Проверка работы:**
   ```bash
   python test_api.py
   ```

## API Эндпоинты

### Health Check
```
GET /health
```
Проверяет состояние API и загруженных моделей.

**Ответ:**
```json
{
  "status": "OK",
  "model_loaded": true,
  "embedding_model_loaded": true
}
```

### Одиночное предсказание
```
POST /predict
```

**Тело запроса:**
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

**Ответ:**
```json
{
  "prediction": 1,
  "probabilities": [0.2, 0.8],
  "track_embedding_dim": 384
}
```

### Пакетное предсказание
```
POST /batch_predict
```

**Тело запроса:**
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

**Ответ:**
```json
{
  "results": [
    {
      "prediction": 0,
      "probabilities": [0.7, 0.3],
      "track": "Bohemian Rhapsody"
    }
  ]
}
```

## Параметры входных данных

| Параметр | Тип | Описание |
|----------|-----|----------|
| `artist` | string | Исполнитель |
| `track` | string | Название трека |
| `decade_of_release` | integer | Декада выпуска (1960, 1970, etc.) |
| `danceability` | float | Танцевальность (0.0-1.0) |
| `energy` | float | Энергичность (0.0-1.0) |
| `key` | integer | Музыкальный ключ (0-11) |
| `loudness` | float | Громкость в дБ |
| `mode` | integer | Лад (0=минор, 1=мажор) |
| `speechiness` | float | Речевость (0.0-1.0) |
| `acousticness` | float | Акустичность (0.0-1.0) |
| `instrumentalness` | float | Инструментальность (0.0-1.0) |
| `liveness` | float | Живость (0.0-1.0) |
| `valence` | float | Позитивность (0.0-1.0) |
| `tempo` | float | Темп в BPM |
| `duration_ms` | integer | Длительность в миллисекундах |
| `time_signature` | integer | Размер (3, 4, 5, etc.) |
| `chorus_hit` | float | Попадание в припев |
| `sections` | integer | Количество секций |

## Тестирование

Запустите тестовый скрипт для проверки всех эндпоинтов:

```bash
python test_api.py
```

## Мониторинг

API включает в себя health check эндпоинт для мониторинга состояния:
- Проверка загрузки ML модели
- Проверка загрузки модели эмбеддингов
- Общий статус приложения

## Технологии

- **FastAPI** - Веб-фреймворк для создания API
- **XGBoost** - Градиентный бустинг для машинного обучения
- **Sentence Transformers** - BERT-эмбеддинги для текста
- **Pandas** - Обработка данных
- **Docker** - Контейнеризация
- **Uvicorn** - ASGI сервер

## Лицензия

Этот проект является частью исследовательского проекта по анализу данных.


