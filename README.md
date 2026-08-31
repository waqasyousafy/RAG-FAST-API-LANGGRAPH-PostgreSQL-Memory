# ReAct Agent with pgvector

A FastAPI-based intelligent agent powered by LangGraph that performs Retrieval-Augmented Generation (RAG) over ingested documents using PostgreSQL with pgvector embeddings.

## Overview

This project implements a ReAct (Reasoning + Acting) agent that can:
- Search through ingested technical manuals, PDFs, and text files
- Reason about queries using LLM capabilities
- Maintain conversation history with PostgreSQL checkpointing
- Authenticate users and manage sessions
- Automatically detect and index new documents in a watched directory

## Features

✨ **AI-Powered RAG**: Search and reason over document collections using semantic embeddings
🔐 **Authentication**: User registration and login with secure password handling
📚 **Multi-Format Support**: Load and process PDF files, text files, and documents
🔄 **Document Watcher**: Automatically detects and indexes new files added to the document directory
💾 **Persistent State**: PostgreSQL-based conversation checkpointing for stateful agent execution
🚀 **Fast Inference**: Uses Groq's fast LLM API for quick responses

## Prerequisites

- Python 3.13+
- PostgreSQL 12+ with pgvector extension
- Groq API key ([get one here](https://console.groq.com))
- Git

## Project Structure

```
.
├── main.py                      # FastAPI application entry point
├── config.py                    # Configuration and environment variables
├── database.py                  # PostgreSQL connection management
├── models.py                    # Pydantic models for requests/responses
├── DocumentChangeHandler.py     # File watcher for document directory
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── routers/
│   ├── auth_router.py          # Authentication endpoints
│   └── aagent_router.py        # Agent chat endpoints
└── services/
    ├── agent_service.py        # Agent initialization and execution
    └── auth_service.py         # User authentication logic
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd reactagent_with_pgvector
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or with uv:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Set up PostgreSQL with pgvector**
   ```bash
   # Install pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;
   
   # Create database
   createdb vectordb
   ```

5. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # LLM Configuration
   LLM=llama-3.1-8b-instant
   GROQ_API_KEY=your_groq_api_key_here
   
   # Database
   DATABASE_URL=postgresql+psycopg://postgres:mysecretpassword@localhost:5432/vectordb
   
   # Document Directory
   DIRECTORY=G:\datato_ingestion
   ```

## Running the Application

1. **Start the server**
   ```bash
   uvicorn main:app --reload
   ```
   
   The API will be available at `http://localhost:8000`

2. **API Documentation**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Authentication

- **POST** `/auth/register` - Register a new user
  ```json
  {
    "username": "user@example.com",
    "password": "securepassword"
  }
  ```

- **POST** `/auth/login` - Login and get authentication token
  ```json
  {
    "username": "user@example.com",
    "password": "securepassword"
  }
  ```

### Agent Chat

- **POST** `/api/chat` - Query the agent (requires authentication)
  ```json
  {
    "query": "What does the documentation say about feature X?",
    "thread_id": "unique_conversation_id"
  }
  ```
  Response:
  ```json
  {
    "response": "Based on the documentation, feature X..."
  }
  ```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM` | `llama-3.1-8b-instant` | Groq LLM model name |
| `GROQ_API_KEY` | Required | Your Groq API key |
| `DATABASE_URL` | `postgresql+psycopg://postgres:mysecretpassword@localhost:5432/vectordb` | PostgreSQL connection string |
| `DIRECTORY` | `G:\datato_ingestion` | Path to document directory for ingestion |

### Application Settings

- **Model**: Configured via `config.py`
- **Vector Store**: pgvector in PostgreSQL
- **Embedding Model**: `all-MiniLM-L6-v2` (from HuggingFace)
- **Collection Name**: `rag_agent_documents`
- **Chunk Size**: 1000 tokens with 0 overlap
- **RAG Search Results**: Top 3 documents with marginal relevance

## Document Ingestion

The application automatically monitors the `DIRECTORY` path for new documents:

1. **Place documents** in the configured `DIRECTORY` path
2. **Supported formats**: `.txt` and `.pdf` files
3. **Automatic indexing**: DocumentChangeHandler watches for file additions and updates the vector store
4. **Chunking**: Documents are split into 1000-token chunks for optimal retrieval

### Adding Documents Manually

Place PDF or TXT files in your `DIRECTORY` path:
```
G:\datato_ingestion\
├── manual1.pdf
├── guide.txt
└── documentation.pdf
```

On next initialization, these documents will be loaded and indexed.

## Architecture

### Agent Flow

```
User Query
    ↓
Authentication
    ↓
LangGraph ReAct Agent
    ├─→ search_documents tool
    │   ├─→ Vector Store Query
    │   └─→ Similarity Search
    └─→ LLM Reasoning
    ↓
Response
```

### Database Schema

- **users**: User accounts and authentication
- **langchain_pg_collection**: Vector store collections
- **langchain_pg_embedding**: Document embeddings and metadata
- **checkpoints**: Agent conversation state (managed by langgraph)

## Development

### Key Components

#### `agent_service.py`
- `initialize_vector_store()`: Loads documents and creates embeddings
- `search_documents()`: LangChain tool for semantic search
- `get_agent_executor()`: Creates and configures the ReAct agent
- `run_agent_workflow()`: Executes agent for user queries

#### `auth_service.py`
- User registration and authentication
- Password hashing and verification
- JWT token generation and validation

#### `DocumentChangeHandler.py`
- File system watcher for new documents
- Automatic vector store updates

## Requirements

See [requirements.txt](requirements.txt) for the complete list. Key dependencies:

- **fastapi**: Web framework
- **langgraph**: Agent orchestration
- **langchain**: LLM framework
- **langchain-postgres**: Vector store
- **langchain-groq**: Groq LLM integration
- **psycopg**: PostgreSQL driver
- **sentence-transformers**: Embedding generation
- **watchdog**: File system monitoring

## Troubleshooting

### Vector Store Initialization Fails
- Ensure PostgreSQL is running and pgvector extension is installed
- Verify `DATABASE_URL` in `.env` file
- Check document directory exists and has readable permissions

### No Documents Found
- Verify documents are in the configured `DIRECTORY` path
- Check file formats are `.txt` or `.pdf`
- Look at application logs for file loading errors
- Restart the application to trigger re-indexing

### Authentication Errors
- Ensure users table is created (happens automatically on startup)
- Verify credentials are correct
- Check JWT token hasn't expired

## Deployment

### Docker (Optional)

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t reactagent-pgvector .
docker run -p 8000:8000 --env-file .env reactagent-pgvector
```

## Performance Optimization

- **Caching**: Agent executor is cached globally to avoid re-initialization
- **Embeddings**: HuggingFace CPU embeddings; consider GPU for high throughput
- **Database**: Ensure PostgreSQL has sufficient resources for large document collections
- **Chunking**: Adjust chunk size in `agent_service.py` for your use case

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

[Specify your license here]

## Authors

- **waqasyousafy** - Initial development

## Support

For issues, questions, or suggestions, please open an issue on the repository.

## Changelog

### v0.1.0
- Initial release
- ReAct agent with document RAG
- User authentication
- PostgreSQL + pgvector integration
- Document watcher
