from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.sql import func
from database import Base

class SignSequence(Base):
    __tablename__ = "sign_sequences"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)  # ชื่อท่ามือ (เช่น Hello)
    frames = Column(JSON)               # เก็บ array ของ landmarks (30 เฟรม)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
