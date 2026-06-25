FROM python:3.13.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# pywin32 is Windows-only and cannot be installed in the Linux container image.
RUN grep -vi '^pywin32==' requirements.txt > /tmp/requirements-docker.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-docker.txt

COPY . .

EXPOSE 8000 10010 10011 10012 10013 10014
