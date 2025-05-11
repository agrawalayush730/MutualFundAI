# database/auth_service/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os


MYSQL_USER = "root"
MYSQL_PASSWORD = "Ayush123#"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB = "auth_service"

# the MySQL connection URL
DATABASE_URL = (
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()
