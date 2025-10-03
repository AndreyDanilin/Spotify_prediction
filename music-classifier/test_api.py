#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы Music Classifier API
"""
import requests
import json
import time

# Базовый URL API
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Тест health check эндпоинта"""
    print("Тестирование health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"Health check успешен: {data}")
            return True
        else:
            print(f"Health check неудачен: {response.status_code}")
            return False
    except Exception as e:
        print(f"Ошибка при health check: {e}")
        return False

def test_single_prediction():
    """Тест одиночного предсказания"""
    print("\nТестирование одиночного предсказания...")
    
    # Тестовые данные
    test_data = {
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
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Предсказание успешно:")
            print(f"   - Предсказанный класс: {result['prediction']}")
            print(f"   - Вероятности: {result['probabilities']}")
            print(f"   - Размерность эмбеддинга: {result['track_embedding_dim']}")
            return True
        else:
            print(f"Ошибка предсказания: {response.status_code}")
            print(f"   Ответ: {response.text}")
            return False
            
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return False

def test_batch_prediction():
    """Тест пакетного предсказания"""
    print("\nТестирование пакетного предсказания...")
    
    # Тестовые данные для пакетной обработки
    test_data = {
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
            },
            {
                "artist": "Michael Jackson",
                "track": "Billie Jean",
                "decade_of_release": 1980,
                "danceability": 0.8,
                "energy": 0.7,
                "key": 0,
                "loudness": -6.8,
                "mode": 1,
                "speechiness": 0.04,
                "acousticness": 0.05,
                "instrumentalness": 0.0,
                "liveness": 0.1,
                "valence": 0.6,
                "tempo": 117.0,
                "duration_ms": 294000,
                "time_signature": 4,
                "chorus_hit": 0.7,
                "sections": 6
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Пакетное предсказание успешно:")
            for i, item in enumerate(result['results']):
                print(f"   Трек {i+1} ({item['track']}):")
                print(f"     - Предсказанный класс: {item['prediction']}")
                print(f"     - Вероятности: {item['probabilities']}")
            return True
        else:
            print(f"Ошибка пакетного предсказания: {response.status_code}")
            print(f"   Ответ: {response.text}")
            return False
            
    except Exception as e:
        print(f"Ошибка при пакетном предсказании: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("Запуск тестов Music Classifier API")
    print("=" * 50)
    
    # Ждем немного, чтобы сервер успел запуститься
    print("Ожидание запуска сервера...")
    time.sleep(3)
    
    # Запуск тестов
    tests = [
        test_health_check,
        test_single_prediction,
        test_batch_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Результаты тестирования: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("Все тесты пройдены успешно!")
        return True
    else:
        print("Некоторые тесты не пройдены")
        return False

if __name__ == "__main__":
    main()
