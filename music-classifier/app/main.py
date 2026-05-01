"""ASGI entrypoint for the Litestar music classifier API."""

from spotify_prediction.api import app, create_app

__all__ = ["app", "create_app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("spotify_prediction.api:app", host="0.0.0.0", port=8000, log_level="info")
