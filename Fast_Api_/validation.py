from pydantic import EmailStr, ValidationError
from entities import RegisterRequest, LoginRequest, FeedbackRequest
import re


# 🔐 Register Validation
def validate_register(data: RegisterRequest):
    if len(data.name.strip()) < 2:
        raise ValueError("Name must be at least 2 characters long.")
    
    if not re.search(r"[A-Za-z]", data.password) or not re.search(r"\d", data.password):
        raise ValueError("Password must include both letters and numbers.")

    if len(data.password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    return data


# 🔑 Login Validation
def validate_login(data: LoginRequest):
    email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    if not data.email or not re.match(email_pattern, data.email.strip().lower()):
        raise ValueError("Invalid email format.")
    
    if not data.password or len(data.password) < 8:
        raise ValueError("Password too short.")
    
    return data


#  Feedback Validation
def validate_feedback(data: FeedbackRequest):
    if len(data.name.strip()) < 2:
        raise ValueError("Name must be at least 2 characters.")

    if len(data.message.strip()) < 10:
        raise ValueError("Feedback message must be at least 10 characters.")
    
    return data
