import os
import math
import datetime
import logging
from typing import List, Optional, Dict
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
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not DATABASE_URL:
    DATABASE_URL = os.environ.get("POSTGRES_URL", "")
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

IS_SQLITE = (not DATABASE_URL) or ("sqlite" in DATABASE_URL)
if IS_SQLITE:
    DATABASE_URL = "sqlite:///./sign_language.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==========================================
# 2. Models
# ==========================================
class SignModel(Base):
    __tablename__ = "signs"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)   # e.g. "ปวดหัว_1", "ปวดหัว_2"
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
# 3. App & CORS
# ==========================================
app = FastAPI(title="Thai Medical Sign AI API", version="3.0")
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

class SequenceInput(BaseModel):
    # list of (step_label, points) pairs sent from frontend
    # e.g. [{"step_label": "ปวดหัว_1", "points": [...]}, {"step_label": "ปวดหัว_2", "points": [...]}]
    steps: List[Dict]


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


def knn_predict_step(query_points, signs, k=5):
    """Predict a single step — returns (label, confidence)."""
    norm_q = normalize_landmarks(query_points)
    distances = [
        (calculate_distance(norm_q, normalize_landmarks(s.landmarks)), s.label)
        for s in signs
    ]
    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]
    if not top_k:
        return "ไม่รู้จักท่าทาง", 0.0
    best_dist = top_k[0][0]
    best_label = Counter(l for _, l in top_k).most_common(1)[0][0]
    confidence = round(1.0 / (1.0 + best_dist * 3.0), 2)
    if best_dist > 0.6:
        return "ไม่รู้จักท่าทาง", confidence
    return best_label, confidence


def parse_label(raw_label: str):
    """
    Parse label like "ปวดหัว_2" → base="ปวดหัว", step=2
    Single-step label like "ขอบคุณ" → base="ขอบคุณ", step=1
    """
    parts = raw_label.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return raw_label, 1


def get_sign_structure(db: Session):
    """
    Returns dict: { base_name: max_steps }
    e.g. {"ปวดหัว": 3, "ขอบคุณ": 1, "เวียนหัว": 2}
    """
    labels = [row[0] for row in db.query(SignModel.label).distinct().all()]
    structure = {}
    for lbl in labels:
        base, step = parse_label(lbl)
        structure[base] = max(structure.get(base, 0), step)
    return structure


# ==========================================
# 6. Endpoints
# ==========================================

@app.get("/")
async def root(db: Session = Depends(get_db)):
    structure = get_sign_structure(db)
    return {
        "status": "online",
        "version": "3.0",
        "database": "sqlite" if IS_SQLITE else "postgresql",
        "total_samples": db.query(SignModel).count(),
        "unique_signs": len(structure),
        "signs": structure,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    """Upload a single landmark frame with label (e.g. 'ปวดหัว_1')."""
    if not payload.label or not payload.points:
        raise HTTPException(status_code=400, detail="Missing label or points")
    try:
        db.add(SignModel(label=payload.label.strip(), landmarks=payload.points))
        db.commit()
        count = db.query(SignModel).filter(SignModel.label == payload.label.strip()).count()
        base, step = parse_label(payload.label.strip())
        return {
            "status": "success",
            "label": payload.label.strip(),
            "base_name": base,
            "step": step,
            "total_for_label": count,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    """
    Predict a single step.
    Returns the raw label (e.g. 'ปวดหัว_1'), base name, step number, and confidence.
    """
    try:
        if not payload.points:
            return {"label": "ไม่พบมือ", "base": "ไม่พบมือ", "step": 0, "confidence": 0}

        # Filter: only predict against the requested step if provided
        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลสอน", "base": "ไม่มีข้อมูลสอน", "step": 0, "confidence": 0}

        label, confidence = knn_predict_step(payload.points, signs)
        base, step = parse_label(label)

        if confidence > 0.3 and label not in ["ไม่รู้จักท่าทาง", "ไม่พบมือ"]:
            db.add(PredictionHistory(label=base, confidence=confidence))
            db.commit()

        return {"label": label, "base": base, "step": step, "confidence": confidence}
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        return {"label": "Error", "base": "Error", "step": 0, "confidence": 0}


@app.post("/predict-step")
async def predict_step(payload: LandmarkInput, db: Session = Depends(get_db)):
    """
    Predict which step label best matches the given landmarks,
    but only among labels that match the expected step number (passed as label field).
    e.g. payload.label = "2" → only compare against _2 labels
    """
    try:
        if not payload.points:
            return {"label": "ไม่พบมือ", "base": "ไม่พบมือ", "step": 0, "confidence": 0}

        target_step = int(payload.label) if payload.label and payload.label.isdigit() else None

        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลสอน", "base": "", "step": 0, "confidence": 0}

        # Filter to only signs at the target step
        if target_step is not None:
            filtered = [s for s in signs if parse_label(s.label)[1] == target_step]
            if not filtered:
                filtered = signs  # fallback to all
        else:
            filtered = signs

        label, confidence = knn_predict_step(payload.points, filtered)
        base, step = parse_label(label)

        return {"label": label, "base": base, "step": step, "confidence": confidence}
    except Exception as e:
        logger.error(f"Predict-step Error: {e}")
        return {"label": "Error", "base": "Error", "step": 0, "confidence": 0}


@app.get("/signs")
def get_signs(db: Session = Depends(get_db)):
    """
    Return all sign names with their step count and sample counts per step.
    e.g. [{"name": "ปวดหัว", "steps": 3, "counts": {"1": 25, "2": 20, "3": 18}}]
    """
    signs = db.query(SignModel).all()
    data: Dict[str, Dict] = {}
    for s in signs:
        base, step = parse_label(s.label)
        if base not in data:
            data[base] = {"name": base, "steps": 0, "counts": {}}
        data[base]["steps"] = max(data[base]["steps"], step)
        key = str(step)
        data[base]["counts"][key] = data[base]["counts"].get(key, 0) + 1

    return sorted(data.values(), key=lambda x: x["name"])


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    signs = db.query(SignModel).all()
    label_counts = {}
    for s in signs:
        label_counts[s.label] = label_counts.get(s.label, 0) + 1
    history = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(50).all()
    structure = get_sign_structure(db)
    return {
        "total_samples": len(signs),
        "unique_signs": len(structure),
        "unique_labels": len(label_counts),
        "labels": [{"label": l, "count": c} for l, c in sorted(label_counts.items())],
        "recent_history": [
            {"label": h.label, "confidence": h.confidence, "created_at": h.created_at}
            for h in history
        ],
    }


@app.get("/history")
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    history = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit).all()
    return [
        {"id": h.id, "label": h.label, "confidence": h.confidence, "created_at": h.created_at}
        for h in history
    ]


@app.delete("/delete/{label:path}")
def delete_label(label: str, db: Session = Depends(get_db)):
    deleted = db.query(SignModel).filter(SignModel.label == label).delete()
    db.commit()
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found")
    return {"status": "success", "deleted_count": deleted, "label": label}


@app.delete("/delete-sign/{base_name:path}")
def delete_sign(base_name: str, db: Session = Depends(get_db)):
    """Delete ALL steps of a sign (e.g. delete 'ปวดหัว' removes ปวดหัว_1, _2, _3)."""
    signs = db.query(SignModel).all()
    to_delete = [s for s in signs if parse_label(s.label)[0] == base_name]
    if not to_delete:
        raise HTTPException(status_code=404, detail=f"Sign '{base_name}' not found")
    for s in to_delete:
        db.delete(s)
    db.commit()
    return {"status": "success", "deleted_count": len(to_delete), "sign": base_name}


@app.delete("/history/clear")
def clear_history(db: Session = Depends(get_db)):
    count = db.query(PredictionHistory).delete()
    db.commit()
    return {"status": "success", "cleared": count}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
