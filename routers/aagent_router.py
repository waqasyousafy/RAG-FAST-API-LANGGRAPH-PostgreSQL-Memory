from fastapi import APIRouter, Depends, HTTPException
from models import AgentQueryRequest, AgentQueryResponse
from services.auth_service import get_current_user
from services.agent_service import run_agent_workflow

router = APIRouter(prefix="/api", tags=["Agent RAG"])

@router.post("/chat", response_model=AgentQueryResponse)
async def chat_with_agent(payload: AgentQueryRequest, user_id: str = Depends(get_current_user)):
    try:
        response_text = run_agent_workflow(payload.query, user_id, payload.thread_id)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))