from fastapi import FastAPI
from contextlib import asynccontextmanager
from services.agent_service import initialize_vector_store, get_agent_executor
from routers import auth_router, aagent_router
from database import get_db_connection  # Import your raw connection function
from services.agent_service import initialize_vector_store, get_agent_executor
from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import get_db_connection  # Import your raw connection function
from services.agent_service import initialize_vector_store, get_agent_executor
from routers import auth_router, aagent_router
from DocumentChangeHandler import start_file_watcher  # assuming you saved it in a utils folder
from config import get_settings  # Import the get_settings function from your config module
@asynccontextmanager
async def lifespan(app: FastAPI):

    setting=get_settings()
    # Create the users table using your raw psycopg connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
          cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
         );
    """)
        conn.commit()
    
    # Preload database embeddings and compile the agent graph upon process startup
    initialize_vector_store()
    start_file_watcher()
    get_agent_executor()
    yield
    print("Application shutdown.")

app = FastAPI(title="Modular LangGraph ReAct Agent API", lifespan=lifespan)

app.include_router(auth_router.router)
app.include_router(aagent_router.router)
