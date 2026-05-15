FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
#   - build-essential, libxml2-dev, libxslt1-dev: needed by lxml (readability-lxml)
#   - curl: used for healthchecks and ad-hoc debugging
#   - libpq5: runtime for any psycopg fallback (asyncpg is pure-python but harmless to keep)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source last for better layer caching.
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
COPY src ./src

EXPOSE 8000

# Run migrations, then start Uvicorn. `alembic upgrade head` is idempotent.
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"]
