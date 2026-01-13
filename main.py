import os
import math
import datetime
import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# ==========================================
# 🔧 Configuration & Logging
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ThaiMedSignAPI")

# ใช้ SQLite เป็นค่าเริ่มต้นเพื่อให้รันได้ทันที (ถ้ามี Postgres ใน Render มันจะเปลี่ยนเองอัตโนมัติ)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./thaimed_sign.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 🗄️ Database Model (60 Frames Sequence)
# ==========================================
class SignSequence(Base):
    __tablename__ = "sign_sequences"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True, nullable=False)
    frames = Column(JSON, nullable=False)  # เก็บ List[List[float]]
    created_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

# ==========================================
# 🔁 Dependency
# ==========================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 🚀 App Lifespan
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database ready")
    except Exception as e:
        logger.error(f"❌ DB init error: {e}")
    yield

app = FastAPI(
    title="Thai Medical Sign AI – Sequence API",
    lifespan=lifespan
)

# ==========================================
# 🌍 CORS (อนุญาตหมดเพื่อให้ Frontend เรียกได้)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📐 Schemas (แยกประเภท Input ให้ชัดเจน)
# ==========================================

# 1. สำหรับตอนสอนท่า (ต้องมี Label)
class SequenceUpload(BaseModel):
    label: str = Field(..., example="ปวดหัว")
    frames: List[List[float]] = Field(
        ..., description="Exactly 60 frames of flattened landmarks"
    )

# 2. สำหรับตอนทำนาย (ส่งมาแค่เฟรม ไม่ต้องส่ง Label)
class SequencePredict(BaseModel):
    frames: List[List[float]] = Field(
        ..., description="Exactly 60 frames of flattened landmarks"
    )

class PredictResponse(BaseModel):
    label: str
    confidence: float

# ==========================================
# 🧮 Sequence Distance Logic (DTW-like)
# ==========================================
def sequence_distance(seq1: List[List[float]], seq2: List[List[float]]) -> float:
    if len(seq1) != len(seq2):
        return float("inf")

    total_dist = 0.0
    valid_frames = 0

    for f1, f2 in zip(seq1, seq2):
        # f1, f2 คือ List[float] ของ Landmarks ใน 1 เฟรม
        if len(f1) != len(f2):
            continue
        
        # Euclidean Distance ระหว่างเฟรมต่อเฟรม
        frame_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))
        total_dist += frame_diff
        valid_frames += 1

    if valid_frames == 0:
        return float("inf")

    return total_dist / valid_frames

# ==========================================
# 📡 API Endpoints
# ==========================================
@app.get("/")
def root():
    return {"status": "online", "mode": "60-frame-sequence"}

@app.get("/dataset")
def dataset(db: Session = Depends(get_db)):
    records = db.query(SignSequence).all()
    return [
        {
            "id": r.id,
            "label": r.label,
            "samples": len(r.frames)
        }
        for r in records
    ]

@app.post("/upload-sequence")
def upload_sequence(payload: SequenceUpload, db: Session = Depends(get_db)):
    # ตรวจสอบจำนวนเฟรม
    if len(payload.frames) != 60:
        raise HTTPException(
            status_code=400,
            detail=f"ต้องการ 60 เฟรม แต่ได้รับ {len(payload.frames)}"
        )

    try:
        record = SignSequence(
            label=payload.label,
            frames=payload.frames
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return {
            "status": "success",
            "id": record.id,
            "label": record.label
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Upload Error: {e}")
        raise HTTPException(500, "Upload failed")

@app.post("/predict-sequence", response_model=PredictResponse)
def predict_sequence(payload: SequencePredict, db: Session = Depends(get_db)):
    records = db.query(SignSequence).all()
    if not records:
        return {"label": "ไม่มีข้อมูล", "confidence": 0.0}

    best_label = "ไม่รู้จัก"
    min_dist = float("inf")

    # เปรียบเทียบกับทุกข้อมูลใน Database (Nearest Neighbor)
    for r in records:
        dist = sequence_distance(payload.frames, r.frames)
        if dist < min_dist:
            min_dist = dist
            best_label = r.label

    # คำนวณ Confidence Score (Distance น้อย = มั่นใจมาก)
    # สูตร: 1 / (1 + distance * sensitivity)
    confidence = 1.0 / (1.0 + min_dist * 2.0)

    # Threshold ตัดเกณฑ์ (ถ้า Distance มากเกินไป ถือว่าไม่รู้จัก)
    # ปรับเลข 1.5 ได้ตามความแม่นยำที่ต้องการ
    if min_dist > 1.5:
        return {
            "label": "ไม่แน่ใจ",
            "confidence": round(confidence, 4)
        }

    return {
        "label": best_label,
        "confidence": round(confidence, 4)
    }
