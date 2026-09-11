FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    fastmcp \
    langchain-community \
    langchain-core \
    langchain-deepseek \
    langchain-groq \
    langchain-huggingface \
    langchain-mcp-adapters==0.3.2 \
    "mcp>=1.9.0,<2.0.0" \
    langchain-ollama \
    langchain-postgres \
    langgraph \
    langgraph-checkpoint-postgres \
    "psycopg[binary]" \
    pydantic \
    pypdf \
    python-dotenv \
    redis \
    sentence-transformers \
    uvicorn \
    watchdog

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