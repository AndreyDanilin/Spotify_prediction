#!/bin/bash


uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info \
    --timeout-keep-alive 120