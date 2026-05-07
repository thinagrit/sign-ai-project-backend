import os
import math
import datetime
import logging
from typing import List, Optional
from collections import Counter

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Database Setup  (Fixed for Render)
# ==========================================

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Render gives "postgres://" but SQLAlchemy needs "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Fallback to POSTGRES_URL if DATABASE_URL missing
if not DATABASE_URL:
    DATABASE_URL = os.environ.get("POSTGRES_URL", "")
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

IS_SQLITE = (not DATABASE_URL) or ("sqlite" in DATABASE_URL)

if IS_SQLITE:
    DATABASE_URL = "sqlite:///./sign_language.db"
    logger.warning("DATABASE_URL not set — using SQLite locally. Add PostgreSQL on Render!")
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    logger.info("Using PostgreSQL database")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==========================================
# 2. Models
# ==========================================
class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    landmarks = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================
# 3. FastAPI & CORS
# ==========================================
app = FastAPI(title="Thai Medical Sign AI API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 4. Schemas
# ==========================================
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float]


# ==========================================
# 5. Helpers
# ==========================================
def normalize_landmarks(points):
    if len(points) < 3:
        return points
    coords = [(points[i], points[i+1], points[i+2]) for i in range(0, len(points), 3)]
    wx, wy, wz = coords[0]
    translated = [(x-wx, y-wy, z-wz) for x,y,z in coords]
    max_dist = max(math.sqrt(x**2+y**2+z**2) for x,y,z in translated) or 1.0
    norm = [(x/max_dist, y/max_dist, z/max_dist) for x,y,z in translated]
    return [v for pt in norm for v in pt]


def calculate_distance(p1, p2):
    if not p1 or not p2:
        return float("inf")
    n = min(len(p1), len(p2))
    return math.sqrt(sum((p1[i]-p2[i])**2 for i in range(n)))


def knn_predict(query_points, signs, k=5):
    norm_q = normalize_landmarks(query_points)
    distances = [(calculate_distance(norm_q, normalize_landmarks(s.landmarks)), s.label) for s in signs]
    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]
    if not top_k:
        return "ไม่รู้จักท่าทาง", 0.0
    best_dist = top_k[0][0]
    best_label = Counter(l for _,l in top_k).most_common(1)[0][0]
    confidence = round(1.0 / (1.0 + best_dist * 3.0), 2)
    if best_dist > 0.6:
        return "ไม่รู้จักท่าทาง", confidence
    return best_label, confidence


# ==========================================
# 6. Endpoints
# ==========================================
@app.get("/")
async def root(db: Session = Depends(get_db)):
    return {
        "status": "online",
        "version": "2.1",
        "database": "sqlite" if IS_SQLITE else "postgresql",
        "total_samples": db.query(SignModel).count(),
        "unique_labels": db.query(SignModel.label).distinct().count(),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    if not payload.label or not payload.points:
        raise HTTPException(status_code=400, detail="Missing label or points")
    try:
        db.add(SignModel(label=payload.label.strip(), landmarks=payload.points))
        db.commit()
        count = db.query(SignModel).filter(SignModel.label == payload.label.strip()).count()
        return {"status": "success", "message": f"Saved '{payload.label}'", "total_for_label": count}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        if not payload.points:
            return {"label": "ไม่พบมือ", "confidence": 0}
        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลสอน", "confidence": 0}
        label, confidence = knn_predict(payload.points, signs)
        if confidence > 0.3 and label not in ["ไม่รู้จักท่าทาง", "ไม่พบมือ"]:
            db.add(PredictionHistory(label=label, confidence=confidence))
            db.commit()
        return {"label": label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        return {"label": "Error", "confidence": 0}


@app.get("/dataset")
def get_dataset(db: Session = Depends(get_db)):
    signs = db.query(SignModel).all()
    return [{"id": s.id, "label": s.label, "landmarks": s.landmarks, "created_at": s.created_at} for s in signs]


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    signs = db.query(SignModel).all()
    label_counts = {}
    for s in signs:
        label_counts[s.label] = label_counts.get(s.label, 0) + 1
    history = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(50).all()
    return {
        "total_samples": len(signs),
        "unique_labels": len(label_counts),
        "labels": [{"label": l, "count": c} for l,c in sorted(label_counts.items())],
        "recent_history": [{"label": h.label, "confidence": h.confidence, "created_at": h.created_at} for h in history],
    }


@app.get("/history")
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    history = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit).all()
    return [{"id": h.id, "label": h.label, "confidence": h.confidence, "created_at": h.created_at} for h in history]


@app.delete("/delete/{label}")
def delete_label(label: str, db: Session = Depends(get_db)):
    deleted = db.query(SignModel).filter(SignModel.label == label).delete()
    db.commit()
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found")
    return {"status": "success", "deleted_count": deleted, "label": label}


@app.delete("/delete/sample/{sample_id}")
def delete_sample(sample_id: int, db: Session = Depends(get_db)):
    sample = db.query(SignModel).filter(SignModel.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    db.delete(sample)
    db.commit()
    return {"status": "success", "deleted_id": sample_id}


@app.delete("/history/clear")
def clear_history(db: Session = Depends(get_db)):
    count = db.query(PredictionHistory).delete()
    db.commit()
    return {"status": "success", "cleared": count}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
