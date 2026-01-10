import os
import math
import datetime
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ThaiMedSignAPI")

# ==========================================
# ⚙️ 1. Database Setup
# ==========================================
DATABASE_URL = os.environ.get("postgresql://thaimed_db_user:7qCAvO14szgLf3FfToANxFq5xOugRxRq@dpg-d5600t6r433s73dslnlg-a/thaimed_db", "sqlite:///./test.db")

# Fix Postgres URL for SQLAlchemy
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite specific arguments
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Models ---
class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    landmarks = Column(JSON)  # Stores List[float]
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 🚀 2. FastAPI Setup with Lifespan
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified.")
    except Exception as e:
        logger.error(f"Database setup error: {e}")
    yield
    # Shutdown: (Cleanup if needed)

app = FastAPI(title="Thai Medical Sign AI API", lifespan=lifespan)

# --- CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production tip: Replace "*" with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Exception Handler (Better than custom middleware) ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error_msg": str(exc)},
    )

# ==========================================
# 📐 3. Schemas & Logic
# ==========================================

class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float] = Field(..., description="Flattened list of landmark coordinates (x, y, z)")

class PredictResponse(BaseModel):
    label: str
    confidence: float

def calculate_distance(p1: List[float], p2: List[float]) -> float:
    """Calculates Euclidean distance between two flattened lists of points."""
    if not p1 or not p2 or len(p1) != len(p2):
        return float('inf')
    
    # Optimization: Use sum generator directly
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# ==========================================
# 📡 4. API Endpoints
# ==========================================

@app.get("/")
def read_root():
    return {"status": "online", "message": "Thai Medical Sign API Ready"}

@app.get("/dataset")
def get_dataset(db: Session = Depends(get_db)):
    """
    Note: Using standard 'def' because Session is synchronous.
    FastAPI will run this in a thread pool to avoid blocking.
    """
    try:
        signs = db.query(SignModel).all()
        return [{"id": s.id, "label": s.label, "landmarks": s.landmarks} for s in signs]
    except Exception as e:
        logger.error(f"Dataset Fetch Error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")

@app.post("/upload")
def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    if not payload.label or not payload.points:
        raise HTTPException(status_code=400, detail="Label and points are required")
    
    try:
        new_sign = SignModel(label=payload.label, landmarks=payload.points)
        db.add(new_sign)
        db.commit()
        db.refresh(new_sign)
        return {"status": "success", "id": new_sign.id, "label": new_sign.label}
    except Exception as e:
        db.rollback()
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลในระบบ", "confidence": 0.0}
        
        best_label = "ไม่รู้จัก"
        min_dist = float('inf')
        
        for item in signs:
            # Check consistency (e.g., one hand vs two hands)
            if len(payload.points) != len(item.landmarks):
                continue

            dist = calculate_distance(payload.points, item.landmarks)
            
            if dist < min_dist:
                min_dist = dist
                best_label = item.label
        
        # Calculate confidence
        # Logic: If distance is 0 (perfect match), confidence is 1.0
        # If distance is high, confidence drops
        confidence = 1.0 / (1.0 + (min_dist * 5.0))
        
        # Threshold Check
        if min_dist > 0.6:  # Adjust this threshold based on testing
            return {"label": "ไม่แน่ใจ", "confidence": round(confidence, 4)}
            
        return {"label": best_label, "confidence": round(confidence, 4)}

    except Exception as e:
        logger.error(f"Predict Error: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")
