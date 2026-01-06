import os
import math
import datetime
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import uvicorn

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# ⚙️ 1. Database Setup (SQLite)
# ==========================================
# ใช้ SQLite สำหรับเก็บข้อมูลพิกัดมือ
DATABASE_URL = "sqlite:///./sign_language.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    landmarks = Column(JSON)  # เก็บ list ของ float
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Schemas ---
class DataInput(BaseModel):
    label: Optional[str] = None
    points: Optional[List[float]] = None   # จากหน้า Data (App.jsx)
    landmark: Optional[List[float]] = None # จากหน้า Predict (App.jsx)

# --- Helper Function ---
def calculate_distance(p1, p2):
    """คำนวณระยะห่างระหว่างชุดพิกัด (Euclidean Distance)"""
    if not p1 or not p2:
        return 1000.0
    
    # คำนวณเฉพาะจุดที่มีคู่กัน (ป้องกันกรณีมือ 1 ข้าง vs 2 ข้าง)
    length = min(len(p1), len(p2))
    if length == 0:
        return 1000.0
        
    sum_sq = 0
    for i in range(length):
        sum_sq += (p1[i] - p2[i]) ** 2
    return math.sqrt(sum_sq)

# ==========================================
# 📡 3. API Endpoints
# ==========================================

@app.get("/")
async def read_root():
    return {"status": "online", "message": "Backend is running"}

@app.post("/upload_video")
@app.post("/upload")
async def upload_data(payload: DataInput, db: Session = Depends(get_db)):
    try:
        # ตรวจสอบว่าข้อมูลมาทางไหน (points หรือ landmark)
        input_points = payload.points if payload.points is not None else payload.landmark
        
        if not payload.label:
            raise HTTPException(status_code=400, detail="กรุณาระบุชื่อท่าทาง (label)")
        if not input_points:
            raise HTTPException(status_code=400, detail="ไม่พบข้อมูลพิกัดมือ (points/landmark)")
        
        new_sign = SignModel(
            label=payload.label, 
            landmarks=input_points
        )
        db.add(new_sign)
        db.commit()
        
        logger.info(f"✅ บันทึกสำเร็จ: {payload.label} (Data size: {len(input_points)})")
        return {"status": "success", "message": f"บันทึกท่า '{payload.label}' เรียบร้อย"}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/predict_realtime")
@app.post("/predict")
async def predict_realtime(payload: DataInput, db: Session = Depends(get_db)):
    try:
        # ดึงพิกัดจากฟิลด์ที่ส่งมา
        current_points = payload.landmark if payload.landmark is not None else payload.points
        
        if not current_points:
            return {"label": "ไม่พบพิกัดมือ", "confidence": 0}

        # ดึงข้อมูลทั้งหมดมาเปรียบเทียบ (แบบง่าย)
        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ยังไม่มีข้อมูลสอน", "confidence": 0}
        
        best_label = "รอตรวจจับ..."
        min_dist = float('inf')
        
        for item in signs:
            dist = calculate_distance(current_points, item.landmarks)
            
            # ปรับจูนค่าความต่างตามจำนวนมือ (ถ้ามี 2 มือระยะห่างจะเยอะขึ้นตามธรรมชาติ)
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
        
        # คำนวณ Confidence (1.0 คือใกล้มาก, 0 คือห่างมาก)
        # ปรับตัวหารตามความเหมาะสมของพิกัด MediaPipe
        confidence = 1.0 / (1.0 + (min_dist * 3.0))
        
        # ตัดเกณฑ์ (Threshold) ถ้าห่างเกินไปแสดงว่าไม่ใช่ท่าที่สอนไว้
        if min_dist > 1.2:
            return {"label": "ไม่รู้จักท่าทางนี้", "confidence": round(confidence, 2)}
            
        return {
            "label": best_label, 
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Predict Error: {str(e)}")
        return {"label": "Error", "confidence": 0}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
