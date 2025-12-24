import os
import math
import datetime
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, exc
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# ⚙️ 1. Database Setup
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    landmarks = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 🚀 2. FastAPI & CORS (The Ultimate Fix)
# ==========================================
app = FastAPI(title="Thai Medical Sign AI API")

# อนุญาตทุกอย่างแบบกว้างที่สุดเพื่อแก้ปัญหา Vercel Dynamic URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตทุกโดเมน
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก Method (GET, POST, OPTIONS)
    allow_headers=["*"],  # อนุญาตทุก Header
    expose_headers=["*"]
)

# Middleware พิเศษสำหรับดักจับ Error และส่ง CORS Header กลับไปเสมอ
@app.middleware("http")
async def cors_handler(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Global Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
            headers={"Access-Control-Allow-Origin": "*"}
        )

# --- Schemas ---
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float]

def calculate_distance(p1, p2):
    if not p1 or not p2 or len(p1) != len(p2):
        return 1000.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# ==========================================
# 📡 3. API Endpoints
# ==========================================

@app.get("/")
async def read_root():
    return {"status": "online", "message": "CORS Fixed"}

@app.get("/dataset")
async def get_dataset(db: Session = Depends(get_db)):
    try:
        signs = db.query(SignModel).all()
        return [{"label": s.label, "landmarks": s.landmarks} for s in signs]
    except Exception as e:
        logger.error(f"Dataset Fetch Error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")

@app.post("/upload")
async def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        if not payload.label or not payload.points:
            raise HTTPException(status_code=400, detail="Data incomplete")
        new_sign = SignModel(label=payload.label, landmarks=payload.points)
        db.add(new_sign)
        db.commit()
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลในระบบ", "confidence": 0}
        
        best_label = "ไม่รู้จัก"
        min_dist = 1000.0
        
        for item in signs:
            # ตรวจสอบขนาดข้อมูล (1 มือ = 63, 2 มือ = 126)
            if len(payload.points) != len(item.landmarks):
                continue
            dist = calculate_distance(payload.points, item.landmarks)
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
                
        confidence = 1.0 / (1.0 + (min_dist * 5.0))
        if min_dist > 0.6:
            return {"label": "ไม่แน่ใจ", "confidence": confidence}
            
        return {"label": best_label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")
