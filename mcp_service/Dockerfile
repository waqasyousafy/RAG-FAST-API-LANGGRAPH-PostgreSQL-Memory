FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastmcp \
    langchain-postgres \
    langchain-huggingface \
    langchain-community \
    sentence-transformers \
    watchdog \
    "psycopg[binary]" \
    python-dotenv \
    langsmith \
    redis \
    "mcp>=1.9.0,<2.0.0"

COPY mcp_service/ ./mcp_service/
COPY config.py .
COPY cache.py .

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["python", "-m", "mcp_service.mcpserver"]