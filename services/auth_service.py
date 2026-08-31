import hashlib
from fastapi import HTTPException, Header
from database import get_db_connection

def authenticate_user(username: str, password: str) -> int:
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT user_id FROM users WHERE username = %s AND password_hash = %s",
                (username, password_hash)
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            return row[0]
    finally:
        conn.close()

async def get_current_user(x_user_id: str = Header(...)) -> str:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (x_user_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=401, detail="Unauthorized user session")
    finally:
        conn.close()
    return x_user_id