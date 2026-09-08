FROM python:3.11-slim

LABEL org.opencontainers.image.title="SSALMP" \
      org.opencontainers.image.description="Simple Search API LLM Summarizer Proxy: OpenWebUI-compatible external search that returns LLM summaries instead of raw page text" \
      org.opencontainers.image.source="https://github.com/vektorprime/ssalmp"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Durable config lives here (mounted as a volume in docker-compose.yml).
VOLUME ["/data"]

EXPOSE 8555
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8555"]
