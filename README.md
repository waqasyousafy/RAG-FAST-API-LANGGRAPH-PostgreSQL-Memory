# ReAct Agent with pgvector

A FastAPI service that answers questions over local PDF and text documents. It combines LangGraph, a Groq chat model, Hugging Face embeddings, PostgreSQL/pgvector, and persistent conversation checkpoints.

## What it does

- Ingests `.pdf` and `.txt` files from a watched directory.
- Stores document embeddings in PostgreSQL with pgvector.
- Retrieves and reranks relevant chunks before the agent answers.
- Preserves conversations by `thread_id` with LangGraph checkpoints.
- Applies response caching, rate limiting, and input checks.
- Exposes interactive API documentation through FastAPI.

## Requirements

- Python 3.13 or newer
- PostgreSQL with the `pgvector` extension
- Redis, used by the response cache and rate limiter
- A Groq API key
- A folder containing the documents to index

## Setup

### 1. Install dependencies

Using `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Using `uv`:

```powershell
uv sync
```

### 2. Prepare PostgreSQL

Create a database and enable pgvector:

```sql
CREATE DATABASE vectordb;
\c vectordb
CREATE EXTENSION IF NOT EXISTS vector;
```

The application creates the `users` table and LangGraph checkpoint tables during startup. The vector store uses the `rag_agent_documents` collection.

### 3. Start Redis

Run Redis locally on its default port, or set `REDIS_URL` to another Redis instance:

```text
redis://localhost:6379/0
```

### 4. Configure the environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
LLM=llama-3.1-8b-instant
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/vectordb
DIRECTORY=C:\path\to\documents
REDIS_URL=redis://localhost:6379/0
```

Optional settings include `GROQ_API_FALLBACK`, `fallback_llm`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT`.

## Run the API

```powershell
uvicorn main:app --reload
```

The service starts on `http://localhost:8000`.

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

On startup, the application initializes the embedding model and vector store, starts the document watcher, and prepares the LangGraph agent. The first startup can take longer while models are downloaded.

## API

All routes use the `/api` prefix.

### Login

`POST /api/login`

```json
{
  "username": "user@example.com",
  "password": "your-password"
}
```

The response contains the numeric `user_id`. There is currently no registration endpoint; users must be provisioned in the `users` table before logging in.

### Chat

`POST /api/chat`

Required header:

```text
X-User-Id: 1
```

Request body:

```json
{
  "query": "What does the documentation say about feature X?",
  "thread_id": "user-1-conversation-1"
}
```

Example with PowerShell:

```powershell
$headers = @{ "X-User-Id" = "1" }
$body = @{ query = "Summarize the installation requirements"; thread_id = "user-1-main" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/chat -Headers $headers -ContentType "application/json" -Body $body
```

### Conversation history

`GET /api/chats/history/{user_id}`

Returns the saved threads and messages associated with a user. This endpoint currently accepts the user ID in the URL and does not require the `X-User-Id` header.

## Document ingestion

Put PDF or UTF-8 text files in the directory configured by `DIRECTORY`:

```text
documents/
├── handbook.pdf
├── setup.txt
└── reference.pdf
```

Files are loaded at startup and new or modified files are indexed by the watchdog observer. Documents are split into chunks, embedded with `all-MiniLM-L6-v2`, stored in pgvector, and searched with maximal marginal relevance before cross-encoder reranking.

If the directory is missing or empty, the service creates it and indexes a placeholder document. For reliable re-indexing after changing the collection or ingestion code, clear the existing vector collection before restarting.

## Project layout

```text
.
├── main.py                    # FastAPI app and startup lifecycle
├── config.py                  # Environment-backed configuration
├── database.py                # PostgreSQL connection helper
├── models.py                  # Request and response models
├── DocumentChangeHandler.py   # PDF/TXT file watcher
├── routers/
│   ├── auth_router.py         # Login route
│   └── aagent_router.py       # Chat and history routes
├── services/
│   ├── agent_service.py       # Ingestion, retrieval, and agent workflow
│   └── auth_service.py        # User lookup and request authentication
├── requirements.txt
└── pyproject.toml
```

## Troubleshooting

**Startup cannot connect to PostgreSQL**

Check that PostgreSQL is running, `DATABASE_URL` is valid, and the `vector` extension is installed.

**Startup cannot connect to Redis**

Start Redis or point `REDIS_URL` at a reachable instance.

**No useful answers are returned**

Confirm that `DIRECTORY` contains readable `.pdf` or `.txt` files. Check the startup logs for loader errors and allow time for the embedding and reranker models to download.

**Login fails**

Confirm that the `users` table contains the requested username. The current authentication service compares the SHA-256 hash of the supplied password with `password_hash`.

## License

No license has been specified yet.
