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
# 1. Database Setup
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./sign_language.db"
    logger.info("Using local SQLite database")
else:
    logger.info("Using remote PostgreSQL database")

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


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
# 2. FastAPI & CORS
# ==========================================
app = FastAPI(title="Thai Medical Sign AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. Schemas
# ==========================================
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float]


class DeleteLabelInput(BaseModel):
    label: str


# ==========================================
# 4. Helper Functions
# ==========================================
def normalize_landmarks(points: List[float]) -> List[float]:
    """
    Normalize landmarks so predictions work regardless of hand position/size.
    - Translate so wrist (point 0) is at origin
    - Scale so the hand fits within a unit box
    """
    if len(points) < 3:
        return points

    coords = [(points[i], points[i+1], points[i+2]) for i in range(0, len(points), 3)]

    # Translate: subtract wrist position (first landmark)
    wrist_x, wrist_y, wrist_z = coords[0]
    translated = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in coords]

    # Scale: divide by max distance from wrist
    max_dist = max(
        math.sqrt(x**2 + y**2 + z**2) for x, y, z in translated
    ) or 1.0

    normalized = [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in translated]

    return [v for pt in normalized for v in pt]


def calculate_distance(p1: List[float], p2: List[float]) -> float:
    """Euclidean distance between two flattened landmark arrays."""
    if not p1 or not p2:
        return float('inf')
    n = min(len(p1), len(p2))
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(n)))


def knn_predict(query_points: List[float], signs, k: int = 5):
    """
    K-Nearest Neighbors prediction with normalized landmarks.
    Returns (label, confidence).
    """
    normalized_query = normalize_landmarks(query_points)

    distances = []
    for sign in signs:
        norm_stored = normalize_landmarks(sign.landmarks)
        dist = calculate_distance(normalized_query, norm_stored)
        distances.append((dist, sign.label))

    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]

    if not top_k:
        return "ไม่รู้จักท่าทาง", 0.0

    best_dist = top_k[0][0]

    # Majority vote among top-k
    label_votes = Counter(label for _, label in top_k)
    best_label = label_votes.most_common(1)[0][0]

    # Distance-based confidence (closer = more confident)
    confidence = 1.0 / (1.0 + best_dist * 3.0)

    # Penalize if best distance is too large
    if best_dist > 0.6:
        return "ไม่รู้จักท่าทาง", round(confidence, 2)

    return best_label, round(confidence, 2)


# ==========================================
# 5. API Endpoints
# ==========================================

@app.get("/")
async def read_root(db: Session = Depends(get_db)):
    total_signs = db.query(SignModel).count()
    unique_labels = db.query(SignModel.label).distinct().count()
    return {
        "status": "online",
        "version": "2.0",
        "total_samples": total_signs,
        "unique_labels": unique_labels,
    }


@app.post("/upload")
async def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    if not payload.label or not payload.points:
        raise HTTPException(status_code=400, detail="Missing label or points")
    try:
        new_sign = SignModel(label=payload.label.strip(), landmarks=payload.points)
        db.add(new_sign)
        db.commit()
        count = db.query(SignModel).filter(SignModel.label == payload.label.strip()).count()
        logger.info(f"Saved: {payload.label} (total for label: {count})")
        return {"status": "success", "message": f"Saved '{payload.label}'", "total_for_label": count}
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

        label, confidence = knn_predict(payload.points, signs, k=5)

        # Save to history if confident
        if confidence > 0.3 and label not in ["ไม่รู้จักท่าทาง", "ไม่พบมือ", "ไม่มีข้อมูลสอน"]:
            history_entry = PredictionHistory(label=label, confidence=confidence)
            db.add(history_entry)
            db.commit()

        return {"label": label, "confidence": confidence}
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        return {"label": "Error", "confidence": 0}


@app.get("/dataset")
def get_dataset(db: Session = Depends(get_db)):
    signs = db.query(SignModel).all()
    return [
        {"id": s.id, "label": s.label, "landmarks": s.landmarks, "created_at": s.created_at}
        for s in signs
    ]


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Return per-label sample counts and recent history."""
    signs = db.query(SignModel).all()
    label_counts = {}
    for s in signs:
        label_counts[s.label] = label_counts.get(s.label, 0) + 1

    stats = [{"label": label, "count": count} for label, count in sorted(label_counts.items())]

    history = (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .limit(50)
        .all()
    )
    history_data = [
        {"label": h.label, "confidence": h.confidence, "created_at": h.created_at}
        for h in history
    ]

    return {
        "total_samples": len(signs),
        "unique_labels": len(label_counts),
        "labels": stats,
        "recent_history": history_data,
    }


@app.get("/history")
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    """Return prediction history."""
    history = (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {"id": h.id, "label": h.label, "confidence": h.confidence, "created_at": h.created_at}
        for h in history
    ]


@app.delete("/delete/{label}")
def delete_label(label: str, db: Session = Depends(get_db)):
    """Delete all training samples for a given label."""
    deleted = db.query(SignModel).filter(SignModel.label == label).delete()
    db.commit()
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found")
    logger.info(f"Deleted {deleted} samples for label: {label}")
    return {"status": "success", "deleted_count": deleted, "label": label}


@app.delete("/delete/sample/{sample_id}")
def delete_sample(sample_id: int, db: Session = Depends(get_db)):
    """Delete a single training sample by ID."""
    sample = db.query(SignModel).filter(SignModel.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    db.delete(sample)
    db.commit()
    return {"status": "success", "deleted_id": sample_id}


@app.delete("/history/clear")
def clear_history(db: Session = Depends(get_db)):
    """Clear all prediction history."""
    count = db.query(PredictionHistory).delete()
    db.commit()
    return {"status": "success", "cleared": count}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
