FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# Container runs as an unprivileged user; /data is writable by that user.
RUN useradd -r -u 10001 appuser && chown -R appuser:appuser /data /app
USER appuser

CMD ["python", "-m", "app.main"]
