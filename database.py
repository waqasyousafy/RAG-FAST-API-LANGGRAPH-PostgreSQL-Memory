import psycopg
from config import DB_CONNECTION

def get_db_connection():
    # Strip SQLAlchemy's driver format so raw psycopg can read it successfully
    clean_url = DB_CONNECTION.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(clean_url)