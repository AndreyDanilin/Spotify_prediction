#!/usr/bin/env python3
"""Small manual smoke test for a running Music Classifier API."""

from __future__ import annotations

import time

import requests

BASE_URL = "http://localhost:8000"


def sample_track(track: str = "Hey Jude", artist: str = "The Beatles", decade: int = 1968) -> dict[str, object]:
    return {
        "track": track,
        "artist": artist,
        "decade_of_release": decade,
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
        "sections": 8,
    }


def test_health_check() -> bool:
    response = requests.get(f"{BASE_URL}/health", timeout=120)
    print("health", response.status_code, response.text)
    return response.status_code == 200


def test_single_prediction() -> bool:
    response = requests.post(f"{BASE_URL}/predict", json=sample_track(), timeout=120)
    print("predict", response.status_code, response.text)
    return response.status_code == 201


def test_batch_prediction() -> bool:
    payload = {"items": [sample_track("Hey Jude"), sample_track("Billie Jean", "Michael Jackson", 1982)]}
    response = requests.post(f"{BASE_URL}/batch_predict", json=payload, timeout=120)
    print("batch_predict", response.status_code, response.text)
    return response.status_code == 201


def main() -> bool:
    print("Waiting for API startup...")
    time.sleep(3)
    checks = [test_health_check, test_single_prediction, test_batch_prediction]
    passed = sum(check() for check in checks)
    print(f"Passed {passed}/{len(checks)} smoke checks")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
