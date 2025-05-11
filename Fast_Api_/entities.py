from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class NameRequest(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    age: int = Field(gt=0, lt=150)

class FeedbackRequest(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    message: Optional[str] = Field(default="No message", max_length=500)

class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, max_length=255)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)

class SIPCreateRequest(BaseModel):
    entities: list
    user_id: int