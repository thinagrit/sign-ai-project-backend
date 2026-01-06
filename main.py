import os
import math
import datetime
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# ⚙️ 1. Database Setup (SQLite)
# ==========================================
DATABASE_URL = "sqlite:///./sign_language.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)      # ชื่อท่าทาง เช่น "ปวดหัว"
    landmarks = Column(JSON)                # เก็บพิกัด [x, y, z, ...] เป็น JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# สร้าง Table ใน Database
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 🚀 2. FastAPI & CORS Configuration
# ==========================================
app = FastAPI(title="Thai Medical Sign AI API")

# ตั้งค่า CORS ให้ยอมรับการเชื่อมต่อจาก Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Schemas ---
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float]

# --- Helper Function ---
def calculate_distance(p1, p2):
    """คำนวณระยะห่างระหว่างจุดพิกัดสองชุด"""
    if not p1 or not p2 or len(p1) != len(p2):
        return 1000.0
    # Euclidean distance
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# ==========================================
# 📡 3. API Endpoints
# ==========================================

@app.get("/")
async def read_root():
    return {"status": "online", "message": "ThaiMed Sign AI Backend is running"}

@app.post("/upload")
async def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    """รับข้อมูลท่าทางและพิกัดเพื่อบันทึกลงฐานข้อมูล"""
    try:
        if not payload.label or not payload.points:
            raise HTTPException(status_code=400, detail="ข้อมูลไม่ครบถ้วน")
        
        new_sign = SignModel(
            label=payload.label, 
            landmarks=payload.points
        )
        db.add(new_sign)
        db.commit()
        
        logger.info(f"บันทึกท่าทางสำเร็จ: {payload.label}")
        return {"status": "success", "message": f"บันทึกท่าทาง '{payload.label}' เรียบร้อย"}
    except Exception as e:
        db.rollback()
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    """ทำนายท่าทางโดยเปรียบเทียบกับข้อมูลที่มีในระบบ (K-Nearest Neighbor Logic)"""
    try:
        # ดึงข้อมูลทั้งหมดจาก Database มาเปรียบเทียบ
        signs = db.query(SignModel).all()
        
        if not signs:
            return {"label": "ไม่มีข้อมูลในฐานข้อมูล", "confidence": 0}
        
        best_label = "ไม่รู้จัก"
        min_dist = 1000.0
        
        # วนลูปหาท่าทางที่ 'ใกล้เคียง' ที่สุด
        for item in signs:
            # ตรวจสอบขนาดข้อมูล (ต้องมีจำนวนจุดเท่ากัน)
            if len(payload.points) != len(item.landmarks):
                continue
                
            dist = calculate_distance(payload.points, item.landmarks)
            
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
        
        # คำนวณค่าความมั่นใจ (Confidence) แบบง่ายๆ จากระยะห่าง
        # ยิ่งระยะห่าง (dist) น้อย ค่าความมั่นใจยิ่งสูง
        confidence = 1.0 / (1.0 + (min_dist * 5.0))
        
        # ถ้าพิกัดต่างกันมากเกินไป ให้ถือว่ายังไม่แน่ใจ
        if min_dist > 0.6:
            return {"label": "รอตรวจจับ...", "confidence": confidence}
            
        return {
            "label": best_label, 
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        raise HTTPException(status_code=500, detail="ระบบประมวลผลผิดพลาด")

@app.get("/dataset/count")
async def get_count(db: Session = Depends(get_db)):
    """ดูจำนวนข้อมูลทั้งหมดที่มีในระบบ"""
    count = db.query(SignModel).count()
    return {"total_records": count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
