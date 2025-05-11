
# Standard Library Imports
import os
from datetime import datetime, timedelta
import traceback
import httpx
# Third-Party Library Imports
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from transformers import pipeline
import bcrypt
import yaml
import logging.config
from dotenv import load_dotenv
import torch
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# Local Application Imports
from database.sip_service.models import SIPRequest
from database.auth_service.models import User
from database.db_init import SessionLocal, get_db
from entities import NameRequest, FeedbackRequest, RegisterRequest, LoginRequest, AnalyzeRequest, SIPCreateRequest
from validation import validate_register, validate_login, validate_feedback
from utils.predictor_utils import get_intent, get_entities
from utils.auth_utils import (
    create_access_token,
    get_current_user,
    decode_token,
    decode_refresh_token,
    create_refresh_token,
)

# Load environment variables
load_dotenv()


with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)


logger = logging.getLogger("app_logger")
logger.info("Logging has been configured.")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info("loging middleware invoked ") 
        logger.info(f"Request: {request.method} {request.url.path} {request.headers}")
        try:
            response = await call_next(request)
            logger.info(f" Response: {request.method} {request.url.path}  - {response.status_code}")
            return response
        except Exception as e:
            logger.error(f" Error on {request.method} {request.url.path} - {str(e)}")
            raise

app=FastAPI()
logger.info("FastAPI application initialized.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # 👈 allow requests from frontend
    allow_credentials=False,
    allow_methods=["*"],  # or just ["POST", "GET"]
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware) 


# -----------------------------
# Loading page
# -----------------------------

@app.post("/greet/")
def greet(name_request: NameRequest):
    logger.debug(f"Received greet request: {name_request}")
    #collection.insert_one(name_request.dict())
    return {"message": f"Hello, {name_request.name}. Your age ({name_request.age}) has been saved!"}


# -----------------------------
# Entities and Intent Prediction
# -----------------------------


@app.post("/analyze-text")
async def analyze_text(data: AnalyzeRequest, current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # Step 1: Intent Classification
        intent_label, confidence = get_intent(data.text)

        # Step 2: Entity Extraction
        entities = get_entities(data.text)

        # Step 3: If not create_sip, return intent/entities directly
        if intent_label != "create_sip":
            return {
                "intent": {
                    "label": intent_label,
                    "confidence": round(confidence, 2),
                },
                "entities": entities,
            }

        # Step 4: Prepare payload for SIP microservice
        sip_payload = {
            "user_id": current_user.id if hasattr(current_user, "id") else 1,  # fallback for testing
            "entities": entities
        }

        # Step 5: Call SIP microservice at port 8002
        async with httpx.AsyncClient() as client:
            sip_response = await client.post("http://localhost:8002/sip-creation", json=sip_payload)

        # Step 6: Handle response from SIP service
        if sip_response.status_code != 200:
            raise HTTPException(status_code=sip_response.status_code, detail=f"SIP service error: {sip_response.text}")

        return sip_response.json()

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}\n{tb}")




# -----------------------------
# User Registration 
# -----------------------------

@app.post("/register/")
def register_user(data: RegisterRequest, db: Session = Depends(get_db)):
    try:
        # Validate data before anything else
    
        validate_register(data)

        existing_user = db.query(User).filter(User.email == data.email).first()
        if existing_user:
            logger.warning(f"Registration attempt with existing email: {data.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists."
            )

        hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(
            name=data.name,
            email=data.email,
            password=hashed_password.decode('utf-8')
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        logger.info(f"✅ New user registered: {new_user.email}")
        return {"message": f"✅ User '{new_user.name}' registered successfully!"}

    except ValueError as e:
        logger.warning(f"❌ Validation error during registration: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f" DB error during registration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database error while saving the user."
        )

    except Exception as e:
        logger.exception(f" Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred.")

# -----------------------------
# User login
# -----------------------------

@app.post("/login/")
def login_user(data: LoginRequest, db: Session = Depends(get_db)):
    try:
        data.email = data.email.strip().lower()
        validate_login(data)

        user = db.query(User).filter(User.email == data.email).first()
        if not user:
            logger.warning(f"Login failed: email not found - {data.email}")
            raise HTTPException(status_code=404, detail="Invalid email or password")

        password_valid = bcrypt.checkpw(data.password.encode('utf-8'), user.password.encode('utf-8'))
        if not password_valid:
            logger.warning(f"Login failed: incorrect password for {data.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")

        access_token = create_access_token({"sub": data.email})
        refresh_token = create_refresh_token({"sub": data.email})

        logger.info(access_token)
        logger.info(refresh_token)
        logger.info(f"✅ Login successful for {data.email}")

        return {
            "message": f"✅ Login successful for {user.name}!",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    except ValueError as e:
        logger.warning(f"❌ Validation error during login: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception(f"❌ Unexpected error during login: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred.")

# -----------------------------
# User Logout
# -----------------------------

@app.post("/logout/")
def logout(request: Request):
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Logout attempted without token.")
        raise HTTPException(status_code=401, detail="Authorization token missing or malformed.")

    token = auth_header.split(" ")[1]

    # We no longer revoke the token in DB — this is purely frontend-managed
    logger.info("User logged out — token discarded client-side.")
    return {"message": "Logged out successfully. Please clear token on client side."}

# -----------------------------
# Token Verification
# -----------------------------

@app.get("/verify-token/")
def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    logger.info("verify_token mehtod invoked")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization token missing or malformed.")

    token = auth_header.split(" ")[1]

    try:
        payload = decode_token(token)
        logger.info(f"Token verified for user: {payload.get('sub')}")
        return True
    except HTTPException as e:
        logger.error(f"Token verification failed: {e.detail}")
        raise e

# -----------------------------
# Token Refresh
# -----------------------------

@app.post("/refresh/")
def refresh_token(request: Request):
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Missing or malformed refresh token.")
        raise HTTPException(status_code=401, detail="Refresh token missing or malformed.")

    refresh_token = auth_header.split(" ")[1]

    try:
        payload = decode_refresh_token(refresh_token)
        user_email = payload.get("sub")

        if not user_email:
            raise HTTPException(status_code=401, detail="Invalid refresh token payload.")

        new_access_token = create_access_token(data={"sub": user_email})
        logger.info(f"Issued new access token for user: {user_email}")

        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }

    except HTTPException as e:
        logger.error(f"Refresh token error: {e.detail}")
        raise e