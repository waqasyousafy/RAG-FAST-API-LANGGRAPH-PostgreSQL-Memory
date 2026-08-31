from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    message: str
    user_id: int

class AgentQueryRequest(BaseModel):
    query: str
    thread_id: str

class AgentQueryResponse(BaseModel):
    response: str