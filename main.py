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
DATABASE_URL = "postgresql://thaimed_db_user:7qCAvO14szgLf3FfToANxFq5xOugRxRq@dpg-d5600t6r433s73dslnlg-a.singapore-postgres.render.com/thaimed_db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    landmarks = Column(JSON)  # เก็บพิกัดทั้งหมดของทั้งสองมือ
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
# ปรับ Schema ให้รองรับชื่อฟิลด์ 'points' สำหรับหน้า Data และ 'landmark' สำหรับหน้า Predict
class DataInput(BaseModel):
    label: Optional[str] = None
    points: Optional[List[float]] = None   # สำหรับ /upload_video
    landmark: Optional[List[float]] = None # สำหรับ /predict_realtime

# --- Helper Function ---
def calculate_distance(p1, p2):
    if not p1 or not p2 or len(p1) != len(p2):
        return 1000.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# ==========================================
# 📡 3. API Endpoints (Fixed Paths)
# ==========================================

@app.get("/")
async def read_root():
    return {"status": "online", "message": "Backend is running and ready for Render deployment"}

# แก้ไข Path ให้ตรงกับ Frontend: /upload_video
@app.post("/upload_video")
@app.post("/upload") # รองรับทั้งสองแบบเพื่อความชัวร์
async def upload_data(payload: DataInput, db: Session = Depends(get_db)):
    try:
        points = payload.points if payload.points else payload.landmark
        if not payload.label or not points:
            raise HTTPException(status_code=400, detail="Missing label or landmarks")
        
        new_sign = SignModel(label=payload.label, landmarks=points)
        db.add(new_sign)
        db.commit()
        
        logger.info(f"Saved label: {payload.label}")
        return {"status": "success", "message": f"Saved '{payload.label}' to database"}
    except Exception as e:
        db.rollback()
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# แก้ไข Path ให้ตรงกับ Frontend: /predict_realtime
@app.post("/predict_realtime")
@app.post("/predict") # รองรับทั้งสองแบบเพื่อความชัวร์
async def predict_realtime(payload: DataInput, db: Session = Depends(get_db)):
    try:
        current_points = payload.landmark if payload.landmark else payload.points
        if not current_points:
            return {"label": "ไม่พบพิกัดมือ", "confidence": 0}

        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ฐานข้อมูลว่างเปล่า", "confidence": 0}
        
        best_label = "ไม่รู้จัก"
        min_dist = 1000.0
        
        for item in signs:
            if len(current_points) != len(item.landmarks):
                continue
            dist = calculate_distance(current_points, item.landmarks)
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
        
        confidence = 1.0 / (1.0 + (min_dist * 5.0))
        
        # กรองผลลัพธ์ที่ห่างเกินไป
        if min_dist > 0.8:
            return {"label": "รอตรวจจับ...", "confidence": confidence}
            
        return {"label": best_label, "confidence": round(confidence, 2)}
        
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        return {"label": "Error", "confidence": 0}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
