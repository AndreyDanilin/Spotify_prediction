FROM python:3.10-slim
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app .

CMD ["sh", "scripts/start.sh"]