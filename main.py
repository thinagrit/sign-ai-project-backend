import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import datetime

# --- Init App ---
app = FastAPI(title="Thai Medical Sign AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Simulation (In-Memory) ---
class Database:
    def __init__(self):
        self.data = []

    def add(self, label, landmarks):
        self.data.append({
            "label": label,
            "landmarks": landmarks,
            "created_at": datetime.datetime.now().isoformat()
        })
    
    def get_all(self):
        return self.data

db = Database()

# เพิ่มข้อมูลตัวอย่าง (1 มือ = 63 จุด, 2 มือ = 126 จุด)
db.add("ตัวอย่าง 1 มือ (นิ่ง)", [0.0]*63) 

# --- Schemas ---
class LandmarkInput(BaseModel):
    label: Optional[str] = None
    points: List[float]

# --- Helper Functions ---
def calculate_distance(points1, points2):
    """คำนวณ Euclidean Distance (ต้องมีจำนวนจุดเท่ากันเท่านั้น)"""
    if len(points1) != len(points2):
        return float('inf') # ถ้าจำนวนมือไม่เท่ากัน ให้ถือว่าไม่เหมือนกันเลย
    
    dist = 0.0
    for i in range(len(points1)):
        dist += (points1[i] - points2[i]) ** 2
    return math.sqrt(dist)

# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Thai Medical Sign AI Backend is Running"}

@app.get("/dataset")
def get_dataset():
    return db.get_all()

@app.post("/upload")
def upload_data(payload: LandmarkInput):
    if not payload.label or not payload.points:
        raise HTTPException(status_code=400, detail="Label and points are required")
    db.add(payload.label, payload.points)
    return {"status": "success", "message": f"Recorded '{payload.label}' with {len(payload.points)} points"}

@app.post("/predict")
def predict(payload: LandmarkInput):
    dataset = db.get_all()
    if not dataset:
        return {"label": "ไม่พบข้อมูล", "confidence": 0.0}

    best_label = "ไม่รู้จัก"
    min_dist = float('inf')

    # วนลูปเทียบกับทุกท่าใน Database
    for item in dataset:
        dist = calculate_distance(payload.points, item["landmarks"])
        if dist < min_dist:
            min_dist = dist
            best_label = item["label"]

    # แปลง Distance เป็น Confidence
    # ปรับจูนค่า 5.0 ได้ตามความเหมาะสม (ค่ายิ่งมาก Confidence ยิ่งลดลงเร็วถ้าไม่เหมือนเป๊ะ)
    confidence = 1.0 / (1.0 + (min_dist * 5.0)) 
    
    if min_dist > 0.5: # ถ้าห่างกันเกินไป (ไม่เหมือนเลย)
         return {"label": "ไม่แน่ใจ", "confidence": confidence}

    return {"label": best_label, "confidence": confidence}