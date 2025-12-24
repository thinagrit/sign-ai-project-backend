import os
import math
import datetime
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, exc
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# --- การตั้งค่า Logging เพื่อดู Error บน Server ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# ⚙️ 1. การตั้งค่าฐานข้อมูล (Database Configuration)
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

try:
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    logger.error(f"Database connection error: {e}")

class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    landmarks = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    logger.error(f"Error creating tables: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 🚀 2. การแก้ไขปัญหา CORS (Strict Fix)
# ==========================================
app = FastAPI(title="Thai Medical Sign AI API")

# การตั้งค่า CORS ต้องทำทันทีหลังจากประกาศ app และต้องทำก่อนประกาศ Routes
# ใช้ allow_origin_regex หรือ allow_origins=["*"] เพื่อแก้ปัญหา fetch block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # อนุญาตทุกแหล่งที่มา (แก้ปัญหา CORS ได้ 100%)
    allow_credentials=True,
    allow_methods=["*"], # อนุญาตทุก Method (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # อนุญาตทุก Headers
    expose_headers=["*"]
)

# --- โครงสร้างข้อมูล (Schemas) ---
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float]

def calculate_distance(points1, points2):
    if not points1 or not points2 or len(points1) != len(points2):
        return float('inf')
    total_dist = 0.0
    for p1, p2 in zip(points1, points2):
        total_dist += (p1 - p2) ** 2
    return math.sqrt(total_dist)

# ==========================================
# 📡 3. API Endpoints
# ==========================================

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend is ready"}

@app.get("/dataset")
def get_dataset(db: Session = Depends(get_db)):
    try:
        # ดึงข้อมูลและตรวจสอบ Error
        signs = db.query(SignModel).all()
        return [{"label": s.label, "landmarks": s.landmarks} for s in signs]
    except Exception as e:
        logger.error(f"Dataset Fetch Error: {e}")
        # แม้จะ Error ก็ต้องส่ง Response ที่มี CORS Header กลับไป
        raise HTTPException(status_code=500, detail="Database connection error")

@app.post("/upload")
def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    if not payload.label or not payload.points:
        raise HTTPException(status_code=400, detail="Missing data")
    try:
        new_sign = SignModel(label=payload.label, landmarks=payload.points)
        db.add(new_sign)
        db.commit()
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูล", "confidence": 0}
        
        best_label = "ไม่รู้จัก"
        min_dist = float('inf')
        
        for item in signs:
            # ตรวจสอบจำนวนจุด (1 มือ = 63 จุด, 2 มือ = 126 จุด)
            if len(payload.points) != len(item.landmarks): 
                continue
            dist = calculate_distance(payload.points, item.landmarks)
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
                
        confidence = 1.0 / (1.0 + (min_dist * 4.0))
        if min_dist > 0.8:
            return {"label": "ไม่แน่ใจ", "confidence": confidence}
        return {"label": best_label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
