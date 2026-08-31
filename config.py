import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv('LLM', 'llama-3.1-8b-instant')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
DOC_DIRECTORY = Path(os.getenv('DIRECTORY', r"G:\datato_ingestion"))
DB_CONNECTION = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg://postgres:mysecretpassword@localhost:5432/vectordb"
)
COLLECTION_NAME = "rag_agent_documents"