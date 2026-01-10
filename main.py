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

DATABASE_URL = os.environ.get(
    "postgresql://thaimed_db_user:7qCAvO14szgLf3FfToANxFq5xOugRxRq@dpg-d5600t6r433s73dslnlg-a/thaimed_db",
    "sqlite:///./thaimed_sign.db"
)

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
    frames = Column(JSON, nullable=False)  # List[List[float]]
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
        logger.info("Database ready")
    except Exception as e:
        logger.error(f"DB init error: {e}")
    yield

app = FastAPI(
    title="Thai Medical Sign AI – Sequence API",
    lifespan=lifespan
)

# ==========================================
# 🌍 CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ❗ Global Error Handler
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

# ==========================================
# 📐 Schemas
# ==========================================
class SequenceInput(BaseModel):
    label: str = Field(..., example="ปวดหัว")
    frames: List[List[float]] = Field(
        ..., description="Exactly 60 frames of flattened landmarks"
    )

class PredictResponse(BaseModel):
    label: str
    confidence: float

# ==========================================
# 🧮 Sequence Distance Logic
# ==========================================
def sequence_distance(seq1: List[List[float]], seq2: List[List[float]]) -> float:
    if len(seq1) != len(seq2):
        return float("inf")

    total = 0.0
    valid = 0

    for f1, f2 in zip(seq1, seq2):
        if len(f1) != len(f2):
            continue
        total += math.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))
        valid += 1

    return total / max(valid, 1)

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
def upload_sequence(payload: SequenceInput, db: Session = Depends(get_db)):
    if len(payload.frames) != 60:
        raise HTTPException(
            status_code=400,
            detail="frames ต้องมี 60 เฟรมเท่านั้น"
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
        logger.error(e)
        raise HTTPException(500, "Upload failed")

@app.post("/predict-sequence", response_model=PredictResponse)
def predict_sequence(payload: SequenceInput, db: Session = Depends(get_db)):
    records = db.query(SignSequence).all()
    if not records:
        return {"label": "ไม่มีข้อมูล", "confidence": 0.0}

    best_label = "ไม่รู้จัก"
    min_dist = float("inf")

    for r in records:
        dist = sequence_distance(payload.frames, r.frames)
        if dist < min_dist:
            min_dist = dist
            best_label = r.label

    confidence = 1 / (1 + min_dist * 3)

    if min_dist > 1.0:
        return {
            "label": "ไม่แน่ใจ",
            "confidence": round(confidence, 4)
        }

    return {
        "label": best_label,
        "confidence": round(confidence, 4)
    }
