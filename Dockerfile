FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    HOSTNAME=0.0.0.0 \
    BRIDGESIGN_DATA_DIR=/tmp/bridgesign-data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        espeak-ng \
        libegl1 \
        libgl1 \
        libgles2 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

CMD ["sh", "-c", "exec gunicorn app:app --worker-class gthread --workers 1 --threads ${GUNICORN_THREADS:-8} --bind 0.0.0.0:${PORT:-10000} --timeout ${GUNICORN_TIMEOUT:-120}"]
