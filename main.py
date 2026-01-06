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
# ⚙️ 1. Database Setup (PostgreSQL for Render)
# ==========================================
# ดึงค่า URL ของฐานข้อมูลจาก Environment Variable ที่ Render กำหนดให้
# หากไม่มี (เช่น รันในเครื่อง) จะใช้ SQLite เป็นค่าเริ่มต้นแทน
DATABASE_URL = os.environ.get("postgresql://thaimed_db_user:7qCAvO14szgLf3FfToANxFq5xOugRxRq@dpg-d5600t6r433s73dslnlg-a/thaimed_db")

# สำหรับ Render PostgreSQL บางกรณี URL จะขึ้นต้นด้วย postgres:// ให้เปลี่ยนเป็น postgresql://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./sign_language.db"
    logger.info("Using local SQLite database")
else:
    logger.info("Using remote PostgreSQL database")

# สร้าง Engine สำหรับเชื่อมต่อ (กรณี SQLite ต้องใช้ check_same_thread=False)
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    landmarks = Column(JSON)  # เก็บ list ของพิกัดมือ
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# สร้าง Table ใน Database (ถ้ายังไม่มี)
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
    return {"status": "online", "message": "Backend is running with PostgreSQL support"}

@app.post("/upload_video")
@app.post("/upload")
async def upload_data(payload: DataInput, db: Session = Depends(get_db)):
    try:
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
        
        logger.info(f"✅ บันทึกสำเร็จ: {payload.label}")
        return {"status": "success", "message": f"บันทึกท่า '{payload.label}' เรียบร้อย"}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/predict_realtime")
@app.post("/predict")
async def predict_realtime(payload: DataInput, db: Session = Depends(get_db)):
    try:
        current_points = payload.landmark if payload.landmark is not None else payload.points
        
        if not current_points:
            return {"label": "ไม่พบพิกัดมือ", "confidence": 0}

        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ยังไม่มีข้อมูลสอน", "confidence": 0}
        
        best_label = "รอตรวจจับ..."
        min_dist = float('inf')
        
        for item in signs:
            dist = calculate_distance(current_points, item.landmarks)
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
        
        confidence = 1.0 / (1.0 + (min_dist * 3.0))
        
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
