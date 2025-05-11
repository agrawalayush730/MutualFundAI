from jose import jwt, JWTError, ExpiredSignatureError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta

# Secret Keys (keep them safe)
SECRET_KEY = "your_access_secret"
REFRESH_SECRET_KEY = "your_refresh_secret"  # Use a different secret for refresh tokens
ALGORITHM = "HS256"

# Expiry durations
ACCESS_TOKEN_EXPIRE_MINUTES = 1
REFRESH_TOKEN_EXPIRE_MINUTES = 15

# FastAPI dependency to auto-extract Bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# -----------------------------
# ACCESS TOKEN FUNCTIONS
# -----------------------------

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Access token has expired")
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid access token")

# -----------------------------
# REFRESH TOKEN FUNCTIONS
# -----------------------------

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, REFRESH_SECRET_KEY, algorithm=ALGORITHM)

def decode_refresh_token(token: str):
    try:
        return jwt.decode(token, REFRESH_SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token has expired")
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid refresh token")

# -----------------------------
# TOKEN DEPENDENCY FOR ROUTES
# -----------------------------

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    payload = decode_token(token)
    email = payload.get("sub")
    if email is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return email
