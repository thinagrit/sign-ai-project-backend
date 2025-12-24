import os
import math
import json
import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- SQLAlchemy Imports (สำหรับเชื่อมต่อ Database) ---
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# ==========================================
# ⚙️ ตั้งค่า Database (Smart Connection)
# ==========================================
# 1. ถ้ามีตัวแปร DATABASE_URL (จาก Render) ให้ใช้ PostgreSQL
# 2. ถ้าไม่มี (เช่น รันในเครื่องตัวเอง) ให้ใช้ SQLite (ไฟล์ test.db)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

# แก้ไข URL ให้ตรงกับ format ที่ SQLAlchemy ต้องการ (Render บางทีส่งมาเป็น postgres://)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# สร้างการเชื่อมต่อ (Engine)
# ตรงนี้แหละที่ระบบจะเรียกใช้ psycopg2-binary อัตโนมัติถ้าเป็น postgresql://
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- สร้างตารางเก็บข้อมูล (Table Model) ---
class SignModel(Base):
    __tablename__ = "signs"
    
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)      # ชื่อท่า (เช่น ปวดหัว)
    landmarks = Column(JSON)                # พิกัดมือ (เก็บเป็น JSON Array)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# สั่งให้สร้างตารางใน Database ถ้ายังไม่มี (Auto Migrate)
Base.metadata.create_all(bind=engine)

# ฟังก์ชันสำหรับดึง DB Session มาใช้และปิดเมื่อเสร็จ
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 🚀 App Setup
# ==========================================
app = FastAPI(title="Thai Medical Sign AI (Persistent DB)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas (ตัวรับส่งข้อมูล) ---
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float]

# --- Helper Functions ---
def calculate_distance(points1, points2):
    """คำนวณ Euclidean Distance (ต้องมีจำนวนจุดเท่ากันเท่านั้น)"""
    # ถ้าจำนวนจุดไม่เท่ากัน (เช่น เทียบ 1 มือ กับ 2 มือ) ถือว่าคนละท่า
    if len(points1) != len(points2):
        return float('inf')
    
    dist = 0.0
    for i in range(len(points1)):
        dist += (points1[i] - points2[i]) ** 2
    return math.sqrt(dist)

# ==========================================
# 📡 Endpoints
# ==========================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Thai Medical Sign AI with Database is Running"}

@app.get("/dataset")
def get_dataset(db: Session = Depends(get_db)):
    # ดึงข้อมูลทั้งหมดจาก Database จริงๆ
    signs = db.query(SignModel).all()
    # ส่งกลับไปให้ Frontend ในรูปแบบ JSON ที่เข้าใจง่าย
    return [{"label": s.label, "landmarks": s.landmarks, "created_at": s.created_at} for s in signs]

@app.post("/upload")
def upload_data(payload: LandmarkInput, db: Session = Depends(get_db)):
    if not payload.label or not payload.points:
        raise HTTPException(status_code=400, detail="Label and points are required")
    
    # บันทึกลง Database (ถาวร)
    new_sign = SignModel(label=payload.label, landmarks=payload.points)
    db.add(new_sign)
    db.commit()
    db.refresh(new_sign)
    
    return {"status": "success", "message": f"Saved '{payload.label}' to Database"}

@app.post("/predict")
def predict(payload: LandmarkInput, db: Session = Depends(get_db)):
    # ดึงข้อมูลครูสอน (Training Data) ทั้งหมดจาก DB มาเทียบ
    signs = db.query(SignModel).all()
    
    if not signs:
        return {"label": "ไม่พบข้อมูลในระบบ", "confidence": 0.0}

    best_label = "ไม่รู้จัก"
    min_dist = float('inf')

    # วนลูปเทียบกับทุกท่าใน Database
    for item in signs:
        # เทียบพิกัดปัจจุบัน กับพิกัดใน DB
        dist = calculate_distance(payload.points, item.landmarks)
        
        if dist < min_dist:
            min_dist = dist
            best_label = item.label

    # แปลง Distance เป็น Confidence (ยิ่งห่างน้อย ยิ่งมั่นใจมาก)
    # ปรับค่า 5.0 ได้ตามความเหมาะสม
    confidence = 1.0 / (1.0 + (min_dist * 5.0)) 
    
    if min_dist > 0.8: # ถ้าห่างกันเกินไป
         return {"label": "ไม่แน่ใจ", "confidence": confidence}

    return {"label": best_label, "confidence": confidence}
