# database/sip_service/sip_schema.py

from pydantic import BaseModel, Field
from typing import Optional


class SIPCreate(BaseModel):
    user_id: int = Field(..., example=1)
    amount: str = Field(..., example="5000")
    fund_name: str = Field(..., example="Axis Bluechip Fund")
    duration: Optional[str] = Field(None, example="1 year")
    start_date: Optional[str] = Field(None, example="July 2025")
    frequency: Optional[str] = Field(None, example="monthly")
