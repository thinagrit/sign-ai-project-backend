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
# ⚙️ 1. Database Setup
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./sign_language.db"
else:
    logger.info("Connecting to PostgreSQL")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
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
    try: yield db
    finally: db.close()

# ==========================================
# 🚀 2. Models & API Setup
# ==========================================
app = FastAPI(title="Thai Medical Sign AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataInput(BaseModel):
    label: Optional[str] = None
    landmark: Optional[List[float]] = None 
    sequence: Optional[List[List[float]]] = None # รองรับ 60 เฟรม

def calculate_distance(p1, p2):
    if not p1 or not p2: return 1000.0
    length = min(len(p1), len(p2))
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(length)))

# ==========================================
# 📡 3. Endpoints
# ==========================================

@app.get("/")
async def read_root():
    return {"status": "online"}

@app.post("/upload_video")
async def upload_video(payload: DataInput, db: Session = Depends(get_db)):
    try:
        # ถ้าส่งมาเป็น Sequence (60 เฟรม) ให้หาค่าเฉลี่ย (Mean) เพื่อเป็นตัวแทนท่าทาง
        if payload.sequence and len(payload.sequence) > 0:
            frame_count = len(payload.sequence)
            point_count = len(payload.sequence[0])
            avg_landmarks = []
            for i in range(point_count):
                avg_val = sum(frame[i] for frame in payload.sequence) / frame_count
                avg_landmarks.append(avg_val)
            pts = avg_landmarks
        else:
            pts = payload.landmark

        if not payload.label or not pts:
            raise HTTPException(status_code=400, detail="Missing label or data")
        
        new_sign = SignModel(label=payload.label, landmarks=pts)
        db.add(new_sign)
        db.commit()
        return {"status": "success", "label": payload.label}
    except Exception as e:
        db.rollback()
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_realtime")
async def predict_realtime(payload: DataInput, db: Session = Depends(get_db)):
    try:
        current_pts = payload.landmark
        if not current_pts:
            return {"label": "ไม่พบพิกัดมือ", "confidence": 0}

        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลสอน", "confidence": 0}
        
        best_label = "รอตรวจจับ..."
        min_dist = float('inf')
        
        for item in signs:
            dist = calculate_distance(current_pts, item.landmarks)
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
        
        confidence = 1.0 / (1.0 + (min_dist * 2.5))
        if min_dist > 1.2:
            return {"label": "ไม่รู้จักท่าทาง", "confidence": round(confidence, 2)}
            
        return {"label": best_label, "confidence": round(confidence, 2)}
    except Exception as e:
        return {"label": "Error", "confidence": 0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
