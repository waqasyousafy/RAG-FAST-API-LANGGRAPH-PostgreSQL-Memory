from fastapi import APIRouter, Depends, HTTPException
from models import AgentQueryRequest, AgentQueryResponse
from services.auth_service import get_current_user
from services.agent_service import run_agent_workflow
from services.agent_service import get_agent_executor
from database import get_db_connection

router = APIRouter(prefix="/api", tags=["Agent RAG"])

@router.post("/chat", response_model=AgentQueryResponse)
async def chat_with_agent(payload: AgentQueryRequest, user_id: str = Depends(get_current_user)):
    try:
        response_text = run_agent_workflow(payload.query, user_id, payload.thread_id)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chats/history/{user_id}")
def get_user_history_by_metadata(user_id: str):
    try:
        # 1. Query distinct thread_ids using the native JSONB metadata column
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT thread_id 
                    FROM checkpoints 
                    WHERE metadata->>'user_id' = %s;
                    """,
                    (user_id,)
                )
                rows = cur.fetchall()

        thread_ids = [row[0] for row in rows]
        agent = get_agent_executor()
        all_threads_history = []

        # 2. Fetch the state natively via the agent for each discovered thread_id
        for thread_id in thread_ids:
            config = {"configurable": {"thread_id": thread_id}}
            state = agent.get_state(config)
            
            messages_list = []
            if state and state.values and "messages" in state.values:
                raw_messages = state.values["messages"]
                
                for msg in raw_messages:
                    if hasattr(msg, "type") and hasattr(msg, "content"):
                        msg_type = msg.type
                        msg_content = msg.content
                    elif isinstance(msg, dict):
                        msg_type = msg.get("type", "human")
                        msg_content = msg.get("content", "")
                    else:
                        msg_type = "human"
                        msg_content = str(msg)
                    
                    role = "user" if msg_type in ("human", "user") else "assistant"
                    messages_list.append({"role": role, "content": msg_content})

            all_threads_history.append({
                "thread_id": thread_id,
                "messages": messages_list
            })

        return {
            "user_id": user_id,
            "threads": all_threads_history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))