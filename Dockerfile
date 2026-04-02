FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for psycopg (Postgres) and bcrypt
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .


# Copy application code
COPY . .

# Ensure checkpoints directory exists
RUN mkdir -p /app/checkpoints

# Default: API only (local dev via docker-compose). Production uses supervisord.
EXPOSE 8000 10000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
