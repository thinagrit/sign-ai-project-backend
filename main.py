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
    logger.info("Using local SQLite database")
else:
    logger.info("Using remote PostgreSQL database")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    landmarks = Column(JSON)  # เก็บ list ของ float (63 หรือ 126 ค่า)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

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

# --- Data Schemas (แบบเฟรมเดียว) ---
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float] # รับอาเรย์ชุดเดียว (เช่น 63 ค่า)

# --- Helper Function ---
def calculate_distance(p1, p2):
    if not p1 or not p2: return 1000.0
    length = min(len(p1), len(p2))
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(length)))

# ==========================================
# 📡 3. API Endpoints
# ==========================================

@app.get("/")
async def read_root():
    return {"status": "online", "mode": "Single Frame"}

@app.post("/upload")
async def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        if not payload.label or not payload.points:
            raise HTTPException(status_code=400, detail="Missing label or points")
        
        new_sign = SignModel(label=payload.label, landmarks=payload.points)
        db.add(new_sign)
        db.commit()
        
        logger.info(f"Saved: {payload.label}")
        return {"status": "success", "message": f"Saved '{payload.label}'"}
    except Exception as e:
        db.rollback()
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        if not payload.points:
            return {"label": "ไม่พบมือ", "confidence": 0}

        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลสอน", "confidence": 0}
        
        best_label = "รอตรวจจับ..."
        min_dist = float('inf')
        
        for item in signs:
            # เทียบระยะห่างระหว่างจุดต่อจุด (Nearest Neighbor)
            dist = calculate_distance(payload.points, item.landmarks)
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
        
        # คำนวณความมั่นใจ (ยิ่งห่างน้อย ยิ่งมั่นใจมาก)
        confidence = 1.0 / (1.0 + (min_dist * 5.0))
        
        if min_dist > 0.8:
            return {"label": "ไม่รู้จักท่าทาง", "confidence": round(confidence, 2)}
            
        return {"label": best_label, "confidence": round(confidence, 2)}
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        return {"label": "Error", "confidence": 0}

@app.get("/dataset")
def get_dataset(db: Session = Depends(get_db)):
    signs = db.query(SignModel).all()
    return [{"label": s.label, "landmarks": s.landmarks, "created_at": s.created_at} for s in signs]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
