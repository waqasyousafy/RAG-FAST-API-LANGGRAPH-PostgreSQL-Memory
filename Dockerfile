FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY main.py .
COPY config.py .
COPY database.py .
COPY cache.py .
COPY rate_limiter.py .
COPY security.py .
COPY models.py .
COPY services/ ./services/
COPY routers/ ./routers/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]