from fastapi import FastAPI
from contextlib import asynccontextmanager

from database import get_db_connection
from services.agent_service import get_agent_executor
from routers import auth_router, aagent_router
from config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    setting = get_settings()

    # Users table — unrelated to RAG/memory, still needed for auth
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

    # Warm up the agent (connects to the MCP server, loads tools) at startup
    # instead of on the first request. Requires mcp_server.py to already be
    # running and reachable — start it before this app.
    await get_agent_executor()

    yield
    print("Application shutdown.")


app = FastAPI(title="Modular LangGraph ReAct Agent API", lifespan=lifespan)

app.include_router(auth_router.router)
app.include_router(aagent_router.router)