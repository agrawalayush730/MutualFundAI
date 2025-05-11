from pydantic import BaseModel, Field, EmailStr
from typing import Optional



class SIPCreateRequest(BaseModel):
    entities: list
    user_id: int