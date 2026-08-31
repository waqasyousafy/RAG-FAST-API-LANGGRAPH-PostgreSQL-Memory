from fastapi import APIRouter
from models import LoginRequest, LoginResponse
from services.auth_service import authenticate_user

router = APIRouter(prefix="/api", tags=["Authentication"])

@router.post("/login", response_model=LoginResponse)
async def login_user(payload: LoginRequest):
    user_id = authenticate_user(payload.username, payload.password)
    return {"message": "Login successful", "user_id": user_id}