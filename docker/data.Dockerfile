# Module 1 data pipeline image: `docker run cricket-ai-data data build ipl`
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CRICKET_AI_DATA_DIR=/app/data \
    CRICKET_AI_REPORTS_DIR=/app/reports

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 1000 app && mkdir -p data reports && chown -R app /app

USER app
VOLUME ["/app/data", "/app/reports"]
ENTRYPOINT ["cricket-ai"]
CMD ["--help"]
