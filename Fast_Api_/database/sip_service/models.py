from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from database.auth_service.sqlalchemy_init import Base

class SIPRequest(Base):
    __tablename__ = "sip_requests"
    __table_args__ = {"schema": "auth_service"}  # ✅ Add this line


    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, nullable=True)  # Optional FK if needed
    amount = Column(String(50), nullable=False)
    fund_name = Column(String(255), nullable=False)
    duration = Column(String(50))
    start_date = Column(String(50))
    frequency = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())
