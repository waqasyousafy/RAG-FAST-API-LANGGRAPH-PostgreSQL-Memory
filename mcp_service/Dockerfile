FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir fastmcp langchain-postgres langchain-huggingface \
    langchain-community sentence-transformers watchdog psycopg[binary] python-dotenv langsmith

COPY mcp_service/ ./mcp_service/
COPY config.py .

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["python", "-m", "mcp_service.mcpserver"]