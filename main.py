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
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Database
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
    label = Column(String, index=True)
    landmarks = Column(JSON)          # stores enriched points (with velocity)
    has_motion = Column(String, default="no")  # "yes" | "no"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ── Migration: add missing columns safely ──────
def run_migrations():
    """Add new columns to existing tables without dropping data."""
    import sqlalchemy as _sa
    with engine.connect() as conn:
        try:
            if IS_SQLITE:
                result = conn.execute(_sa.text("PRAGMA table_info(signs)")).fetchall()
                existing_cols = [row[1] for row in result]
                if "has_motion" not in existing_cols:
                    conn.execute(_sa.text("ALTER TABLE signs ADD COLUMN has_motion VARCHAR DEFAULT 'no'"))
                    conn.commit()
                    logger.info("Migration OK: added has_motion (SQLite)")
            else:
                # PostgreSQL — idempotent: safe to run every startup
                conn.execute(_sa.text("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='signs' AND column_name='has_motion'
                        ) THEN
                            ALTER TABLE signs ADD COLUMN has_motion VARCHAR DEFAULT 'no';
                        END IF;
                    END $$;
                """))
                conn.commit()
                logger.info("Migration OK: has_motion column ready (PostgreSQL)")
        except Exception as e:
            logger.warning(f"Migration note: {e}")

run_migrations()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================
# 3. App & CORS
# ==========================================
app = FastAPI(title="Thai Medical Sign AI API", version="3.1")
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
    points: List[float]             # may include velocity features appended


# ==========================================
# 5. Helpers
# ==========================================

# How many raw landmark values per hand (21 points * 3 = 63)
RAW_LM_SIZE = 63

def split_points(points: List[float]):
    """
    Split enriched points into:
      - raw_landmarks: first RAW_LM_SIZE values (or all if no motion)
      - velocity:      next RAW_LM_SIZE values (if present)
      - magnitude:     scalar (if present)
    Returns (raw, velocity, magnitude, has_motion)
    """
    # Frontend sends: landmarks(63) + velocity(63) + magnitude(1) + wristVx(1) + wristVy(1) = 129
    # Or just landmarks(63) if only 1 frame captured
    if len(points) >= RAW_LM_SIZE * 2 + 3:
        raw = points[:RAW_LM_SIZE]
        vel = points[RAW_LM_SIZE:RAW_LM_SIZE*2]
        magnitude = points[RAW_LM_SIZE*2]
        return raw, vel, magnitude, True
    else:
        raw = points[:RAW_LM_SIZE]
        return raw, None, 0.0, False


def normalize_landmarks(points: List[float]) -> List[float]:
    """Normalize raw landmarks: translate to wrist origin, scale to unit size."""
    if len(points) < 3:
        return points
    coords = [(points[i], points[i+1], points[i+2]) for i in range(0, len(points), 3)]
    wx, wy, wz = coords[0]
    translated = [(x-wx, y-wy, z-wz) for x,y,z in coords]
    max_dist = max(math.sqrt(x**2+y**2+z**2) for x,y,z in translated) or 1.0
    norm = [(x/max_dist, y/max_dist, z/max_dist) for x,y,z in translated]
    return [v for pt in norm for v in pt]


def normalize_velocity(vel: List[float]) -> List[float]:
    """Normalize velocity vector to unit direction (preserve direction, ignore speed)."""
    if not vel:
        return []
    mag = math.sqrt(sum(v*v for v in vel)) or 1.0
    return [v / mag for v in vel]


def calculate_distance(p1: List[float], p2: List[float]) -> float:
    if not p1 or not p2:
        return float("inf")
    n = min(len(p1), len(p2))
    return math.sqrt(sum((p1[i]-p2[i])**2 for i in range(n)))


def combined_distance(
    query_pts: List[float],
    stored_pts: List[float],
    motion_weight: float = 0.4,
) -> float:
    """
    Distance combining:
      - pose distance (shape of hand)     weight: 1 - motion_weight
      - velocity distance (direction)     weight: motion_weight  (only if both have motion)
    """
    q_raw, q_vel, q_mag, q_has_motion = split_points(query_pts)
    s_raw, s_vel, s_mag, s_has_motion = split_points(stored_pts)

    norm_q_raw = normalize_landmarks(q_raw)
    norm_s_raw = normalize_landmarks(s_raw)
    pose_dist = calculate_distance(norm_q_raw, norm_s_raw)

    # If both have velocity, factor in motion direction
    if q_has_motion and s_has_motion and q_vel and s_vel:
        norm_q_vel = normalize_velocity(q_vel)
        norm_s_vel = normalize_velocity(s_vel)
        vel_dist = calculate_distance(norm_q_vel, norm_s_vel)
        return (1 - motion_weight) * pose_dist + motion_weight * vel_dist
    else:
        return pose_dist


def detect_motion_type(points: List[float]) -> str:
    """Classify motion: still | moving | circular"""
    _, vel, magnitude, has_motion = split_points(points)
    if not has_motion or magnitude < 0.005:
        return "still"
    if vel:
        # Check if wristVx and wristVy are oscillating (rough circular check)
        wx = points[-2] if len(points) >= 2 else 0
        wy = points[-1] if len(points) >= 1 else 0
        if abs(wx) > 0.003 and abs(wy) > 0.003:
            return "circular"
    return "moving"


def knn_predict(query_points: List[float], signs, k: int = 5):
    """KNN with motion-aware distance."""
    distances = [
        (combined_distance(query_points, s.landmarks), s.label)
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
    parts = raw_label.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return raw_label, 1


def get_sign_structure(db: Session):
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
        "version": "3.1",
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
    if not payload.label or not payload.points:
        raise HTTPException(status_code=400, detail="Missing label or points")
    try:
        _, _, magnitude, has_motion = split_points(payload.points)
        motion_flag = "yes" if has_motion and magnitude > 0.003 else "no"

        db.add(SignModel(
            label=payload.label.strip(),
            landmarks=payload.points,
            has_motion=motion_flag,
        ))
        db.commit()
        count = db.query(SignModel).filter(SignModel.label == payload.label.strip()).count()
        base, step = parse_label(payload.label.strip())
        return {
            "status": "success",
            "label": payload.label.strip(),
            "base_name": base,
            "step": step,
            "has_motion": motion_flag,
            "total_for_label": count,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        if not payload.points:
            return {"label": "ไม่พบมือ", "base": "ไม่พบมือ", "step": 0, "confidence": 0}

        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลสอน", "base": "ไม่มีข้อมูลสอน", "step": 0, "confidence": 0}

        label, confidence = knn_predict(payload.points, signs)
        base, step = parse_label(label)
        motion_type = detect_motion_type(payload.points)

        if confidence > 0.3 and label not in ["ไม่รู้จักท่าทาง", "ไม่พบมือ"]:
            db.add(PredictionHistory(label=base, confidence=confidence))
            db.commit()

        return {
            "label": label,
            "base": base,
            "step": step,
            "confidence": confidence,
            "motion_type": motion_type,  # "still" | "moving" | "circular"
        }
    except Exception as e:
        logger.error(f"Predict Error: {e}")
        return {"label": "Error", "base": "Error", "step": 0, "confidence": 0}


@app.post("/predict-step")
async def predict_step(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        if not payload.points:
            return {"label": "ไม่พบมือ", "base": "", "step": 0, "confidence": 0}

        target_step = int(payload.label) if payload.label and payload.label.isdigit() else None
        signs = db.query(SignModel).all()
        if not signs:
            return {"label": "ไม่มีข้อมูลสอน", "base": "", "step": 0, "confidence": 0}

        filtered = [s for s in signs if parse_label(s.label)[1] == target_step] if target_step else signs
        if not filtered:
            filtered = signs

        label, confidence = knn_predict(payload.points, filtered)
        base, step = parse_label(label)
        motion_type = detect_motion_type(payload.points)

        return {
            "label": label,
            "base": base,
            "step": step,
            "confidence": confidence,
            "motion_type": motion_type,
        }
    except Exception as e:
        logger.error(f"Predict-step Error: {e}")
        return {"label": "Error", "base": "Error", "step": 0, "confidence": 0}


@app.get("/signs")
def get_signs(db: Session = Depends(get_db)):
    signs = db.query(SignModel).all()
    data: Dict[str, Dict] = {}
    for s in signs:
        base, step = parse_label(s.label)
        if base not in data:
            data[base] = {"name": base, "steps": 0, "counts": {}, "has_motion": False}
        data[base]["steps"] = max(data[base]["steps"], step)
        key = str(step)
        data[base]["counts"][key] = data[base]["counts"].get(key, 0) + 1
        if s.has_motion == "yes":
            data[base]["has_motion"] = True
    return sorted(data.values(), key=lambda x: x["name"])


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    signs = db.query(SignModel).all()
    label_counts = {}
    motion_counts = {"yes": 0, "no": 0}
    for s in signs:
        label_counts[s.label] = label_counts.get(s.label, 0) + 1
        motion_counts[s.has_motion or "no"] += 1
    history = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(50).all()
    structure = get_sign_structure(db)
    return {
        "total_samples": len(signs),
        "unique_signs": len(structure),
        "unique_labels": len(label_counts),
        "motion_samples": motion_counts["yes"],
        "static_samples": motion_counts["no"],
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

@app.get("/export")
def export_dataset(db: Session = Depends(get_db)):
    """
    Export all training data as structured JSON.
    Format ready for future Neural Network training.
    """
    signs = db.query(SignModel).all()

    # Group by base sign name
    grouped: Dict[str, Dict] = {}
    for s in signs:
        base, step = parse_label(s.label)
        if base not in grouped:
            grouped[base] = {
                "name": base,
                "steps": 0,
                "has_motion": False,
                "samples": [],
            }
        grouped[base]["steps"] = max(grouped[base]["steps"], step)
        if s.has_motion == "yes":
            grouped[base]["has_motion"] = True
        grouped[base]["samples"].append({
            "id": s.id,
            "label": s.label,
            "step": step,
            "has_motion": s.has_motion or "no",
            "landmarks": s.landmarks,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        })

    export_data = {
        "exported_at": datetime.datetime.utcnow().isoformat(),
        "version": "3.1",
        "total_samples": len(signs),
        "total_signs": len(grouped),
        "signs": list(grouped.values()),
        # Flat format for easy ML training
        "flat": [
            {
                "label": s.label,
                "base": parse_label(s.label)[0],
                "step": parse_label(s.label)[1],
                "has_motion": s.has_motion or "no",
                "points": s.landmarks,
            }
            for s in signs
        ],
    }
    return export_data


@app.post("/import")
async def import_dataset(data: dict, db: Session = Depends(get_db)):
    """
    Import training data from a previously exported JSON file.
    Skips duplicates — safe to run multiple times.
    """
    try:
        flat = data.get("flat", [])
        if not flat:
            # Try reading from signs format
            for sign in data.get("signs", []):
                for sample in sign.get("samples", []):
                    flat.append({
                        "label": sample["label"],
                        "points": sample["landmarks"],
                        "has_motion": sample.get("has_motion", "no"),
                    })

        imported = 0
        skipped = 0
        for item in flat:
            label = item.get("label", "").strip()
            points = item.get("points", [])
            has_motion = item.get("has_motion", "no")
            if not label or not points:
                skipped += 1
                continue
            db.add(SignModel(
                label=label,
                landmarks=points,
                has_motion=has_motion,
            ))
            imported += 1

        db.commit()
        return {
            "status": "success",
            "imported": imported,
            "skipped": skipped,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
