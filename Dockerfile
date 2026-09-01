# Production Multi-Stage / Minimal Python Dockerfile
FROM python:3.10-slim

# Set environment variables for performance and clean logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (build-essential / curl for healthcheck if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for efficient Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY audio_buffer.py /app/audio_buffer.py
COPY vad_engine.py /app/vad_engine.py
COPY model_helper.py /app/model_helper.py
COPY server.py /app/server.py
COPY test_client.py /app/test_client.py
COPY static /app/static

# Pre-generate the ONNX model inside the image for zero cold-start latency
RUN python model_helper.py --output /app/spoof_detector.onnx --test

# Expose HTTP & WebSocket port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run FastAPI with Uvicorn dynamically binding to Render's $PORT
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
