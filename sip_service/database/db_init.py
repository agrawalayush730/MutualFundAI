# --- MongoDB Setup ---
from pymongo import MongoClient
from dotenv import load_dotenv
import os
# --- MySQL (SQLAlchemy) Setup ---
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database.auth_service.models import Base
from database.auth_service.sqlalchemy_init import DATABASE_URL  

load_dotenv("cred.env")

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["fastapi_app"]

# Mongo collections

feedback_collection = mongo_db["feedback"]


# SQLAlchemy engine + session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI dependency injection for SQLAlchemy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
