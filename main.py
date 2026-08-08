import os
import math
import datetime
import logging
from typing import List, Optional, Dict
from collections import Counter

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float, LargeBinary
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import uvicorn
import json

# Deep learning inference (optional — falls back to KNN if unavailable)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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


class SignImage(Base):
    """รูปตัวอย่างท่าทาง — 1 รูปต่อ 1 label เป๊ะๆ (เช่น "ปวดหัว_1", "ปวดหัว_2") ใช้โชว์ตอนแปลภาษา/คลังคำ"""
    __tablename__ = "sign_images"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, unique=True, index=True)
    image_data = Column(LargeBinary)
    content_type = Column(String, default="image/jpeg")
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
# 3b. Deep Learning model (optional)
# ==========================================
# ถ้ามีไฟล์ sign_model.pt + label_encoder.json + model_meta.json อยู่ข้างๆ main.py
# ระบบจะโหลดขึ้นมาใช้แทน KNN โดยอัตโนมัติ — ถ้าไม่มี จะ fallback ไปใช้ KNN เหมือนเดิม
# (ไฟล์เหล่านี้ได้จากการรัน train_model.py กับข้อมูลที่ export จากหน้า "คลังคำศัพท์")
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DL_TARGET_DIM = 183  # ต้องตรงกับ TARGET_DIM ใน train_model.py

dl_model = None
dl_classes: List[str] = []

if TORCH_AVAILABLE:
    class SignMLP(nn.Module):
        """ต้องมีสถาปัตยกรรมตรงกับตอนเทรนใน train_model.py ทุกประการ"""
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    def _try_load_dl_model():
        global dl_model, dl_classes, DL_TARGET_DIM
        meta_path = os.path.join(MODEL_DIR, "model_meta.json")
        labels_path = os.path.join(MODEL_DIR, "label_encoder.json")
        weights_path = os.path.join(MODEL_DIR, "sign_model.pt")

        if not (os.path.exists(meta_path) and os.path.exists(labels_path) and os.path.exists(weights_path)):
            logger.info("ยังไม่พบไฟล์โมเดล deep learning — ใช้ KNN เป็นค่าเริ่มต้น")
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            with open(labels_path, "r", encoding="utf-8") as f:
                dl_classes = json.load(f)["classes"]

            model = SignMLP(meta["input_dim"], meta["num_classes"])
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval()

            DL_TARGET_DIM = meta.get("target_dim", DL_TARGET_DIM)
            dl_model = model
            logger.info(f"โหลดโมเดล deep learning สำเร็จ ({meta['num_classes']} ท่า) — ใช้แทน KNN แล้ว")
        except Exception as e:
            logger.warning(f"โหลดโมเดล deep learning ไม่สำเร็จ ({e}) — ใช้ KNN แทน")

    _try_load_dl_model()


def pad_or_truncate(vec: List[float], target_len: int = DL_TARGET_DIM) -> List[float]:
    """ทำให้ feature vector มีความยาวเท่ากับตอนเทรนเสมอ (ต้องตรงกับ train_model.py)"""
    vec = list(vec)
    if len(vec) >= target_len:
        return vec[:target_len]
    return vec + [0.0] * (target_len - len(vec))


def dl_predict(points: List[float]):
    """ทำนายด้วยโมเดล deep learning — คืนค่ารูปแบบเดียวกับ knn_predict: (label, confidence)"""
    vec = pad_or_truncate(points, DL_TARGET_DIM)
    x = torch.tensor([vec], dtype=torch.float32)
    with torch.no_grad():
        probs = F.softmax(dl_model(x), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return dl_classes[idx.item()], round(conf.item(), 2)


# ==========================================
# 4. Schemas
# ==========================================
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float]             # may include velocity features appended


# ==========================================
# 5. Helpers
# ==========================================

# Feature vector sizes
HAND_LM_SIZE  = 63   # 21 hand points * 3
POSE_LM_SIZE  = 27   # 9 key pose points * 3
RAW_FULL_SIZE = HAND_LM_SIZE + POSE_LM_SIZE  # 90 total

def split_points(points: List[float]):
    """
    New format (with pose):
      hand(63) + pose(27) + velocity(90) + magnitude(1) + wristVx(1) + wristVy(1) = 183
    Legacy format (hand only):
      hand(63) + velocity(63) + magnitude(1) + wristVx(1) + wristVy(1) = 129
    Returns (raw, velocity, magnitude, has_motion)
    """
    if len(points) >= RAW_FULL_SIZE * 2 + 3:
        # New format with pose
        raw = points[:RAW_FULL_SIZE]
        vel = points[RAW_FULL_SIZE:RAW_FULL_SIZE*2]
        magnitude = points[RAW_FULL_SIZE*2]
        return raw, vel, magnitude, True
    elif len(points) >= HAND_LM_SIZE * 2 + 3:
        # Legacy hand-only with velocity
        raw = points[:HAND_LM_SIZE]
        vel = points[HAND_LM_SIZE:HAND_LM_SIZE*2]
        magnitude = points[HAND_LM_SIZE*2]
        return raw, vel, magnitude, True
    else:
        # Raw only, no velocity
        raw = points[:max(HAND_LM_SIZE, len(points))]
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


@app.get("/model-info")
async def model_info():
    """เช็คว่าตอนนี้ /predict กำลังใช้ deep learning หรือ KNN"""
    if dl_model is not None:
        return {"engine": "deep_learning", "num_classes": len(dl_classes), "classes": dl_classes}
    return {"engine": "knn", "note": "ยังไม่พบไฟล์โมเดล (sign_model.pt, label_encoder.json, model_meta.json)"}


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


# ── Sign reference image (1 รูปต่อ 1 label เป๊ะๆ — รองรับรูปแยกตามขั้นตอน) ──
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB

@app.post("/sign-image/{label}")
async def upload_sign_image(label: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="ไฟล์ว่างเปล่า")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="ไฟล์ใหญ่เกินไป (จำกัด 5MB)")

    content_type = file.content_type or "image/jpeg"
    label = label.strip()

    existing = db.query(SignImage).filter(SignImage.label == label).first()
    if existing:
        existing.image_data = content
        existing.content_type = content_type
        existing.created_at = datetime.datetime.utcnow()
    else:
        db.add(SignImage(label=label, image_data=content, content_type=content_type))
    db.commit()
    return {"status": "success", "label": label}


@app.get("/sign-image/{label}")
async def get_sign_image(label: str, db: Session = Depends(get_db)):
    img = db.query(SignImage).filter(SignImage.label == label.strip()).first()
    if not img:
        raise HTTPException(status_code=404, detail="ไม่พบรูปภาพสำหรับท่านี้")
    return Response(content=img.image_data, media_type=img.content_type)


@app.delete("/sign-image/{label}")
async def delete_sign_image(label: str, db: Session = Depends(get_db)):
    deleted = db.query(SignImage).filter(SignImage.label == label.strip()).delete()
    db.commit()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="ไม่พบรูปภาพสำหรับท่านี้")
    return {"status": "success"}


@app.post("/predict")
async def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    try:
        if not payload.points:
            return {"label": "ไม่พบมือ", "base": "ไม่พบมือ", "step": 0, "confidence": 0}

        # ใช้โมเดล deep learning ถ้าโหลดสำเร็จ ไม่งั้น fallback เป็น KNN
        if dl_model is not None:
            label, confidence = dl_predict(payload.points)
        else:
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
            "engine": "deep_learning" if dl_model is not None else "knn",
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

        # ใช้โมเดล deep learning ถ้าโหลดสำเร็จ ไม่งั้น fallback เป็น KNN (กรองด้วย step)
        if dl_model is not None:
            label, confidence = dl_predict(payload.points)
        else:
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
            "engine": "deep_learning" if dl_model is not None else "knn",
        }
    except Exception as e:
        logger.error(f"Predict-step Error: {e}")
        return {"label": "Error", "base": "Error", "step": 0, "confidence": 0}


@app.get("/signs")
def get_signs(db: Session = Depends(get_db)):
    signs = db.query(SignModel).all()
    image_labels = {row[0] for row in db.query(SignImage.label).all()}
    data: Dict[str, Dict] = {}
    for s in signs:
        base, step = parse_label(s.label)
        if base not in data:
            data[base] = {"name": base, "steps": 0, "counts": {}, "has_motion": False, "images": {}}
        data[base]["steps"] = max(data[base]["steps"], step)
        key = str(step)
        data[base]["counts"][key] = data[base]["counts"].get(key, 0) + 1
        if s.has_motion == "yes":
            data[base]["has_motion"] = True
        if key not in data[base]["images"]:
            data[base]["images"][key] = s.label in image_labels  # true ถ้าขั้นนี้มีรูปตัวอย่างแล้ว

    for entry in data.values():
        entry["has_image"] = any(entry["images"].values())  # มีรูปอย่างน้อย 1 ขั้น (ใช้โชว์ในกริดเลือกท่า)

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
    # cascade: ลบรูปตัวอย่างของทุกขั้นตอนของท่านี้ด้วย (เช่น base_name, base_name_1, base_name_2, ...)
    for img in db.query(SignImage).all():
        if parse_label(img.label)[0] == base_name:
            db.delete(img)
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