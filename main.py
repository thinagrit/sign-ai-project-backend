import os
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
import tensorflow as tf
from pydantic import BaseModel

# Import database modules
from database import engine, get_db, Base
from models import SignSequence

# สร้างตารางใน Database อัตโนมัติ (ถ้ายังไม่มี)
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model (เหมือนเดิม)
MODEL_PATH = "model.h5"
model = None
CLASSES = []

def load_ai_model():
    global model, CLASSES
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            # ดึงรายชื่อ Class จาก Database แทนการอ่านโฟลเดอร์
            db = next(get_db())
            results = db.query(SignSequence.label).distinct().all()
            CLASSES = sorted([r[0] for r in results])
            print(f"✅ Model loaded. Classes: {CLASSES}")
        else:
            print("⚠️ Model not found.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# เรียกโหลดตอนเริ่ม
load_ai_model()

# Pydantic Schemas
class CollectRequest(BaseModel):
    label: str
    frames: list 

class PredictRequest(BaseModel):
    frames: list

@app.get("/")
def read_root():
    return {"status": "Database Connected API Running"}

@app.get("/dataset")
def get_dataset_stats(db: Session = Depends(get_db)):
    """ดึงจำนวนข้อมูลจาก Database Group ตาม Label"""
    results = db.query(SignSequence.label, func.count(SignSequence.id)).group_by(SignSequence.label).all()
    stats = [{"label": r[0], "count": r[1]} for r in results]
    return stats

@app.post("/collect")
def collect_data(data: CollectRequest, db: Session = Depends(get_db)):
    """บันทึกข้อมูลลง PostgreSQL"""
    try:
        # สร้าง Object ใหม่
        new_sequence = SignSequence(
            label=data.label,
            frames=data.frames # SQL Alchemy จะแปลง List เป็น JSON ให้เอง
        )
        
        db.add(new_sequence)
        db.commit()
        db.refresh(new_sequence)
        
        return {"message": f"Saved {data.label} to DB ID: {new_sequence.id}"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(data: PredictRequest):
    global model, CLASSES
    
    if model is None:
        return {"label": "Model not ready", "confidence": 0.0}

    try:
        input_data = np.array(data.frames)
        if input_data.shape[0] < 10: 
             return {"label": "...", "confidence": 0.0}
             
        input_data = np.expand_dims(input_data, axis=0)
        
        prediction = model.predict(input_data, verbose=0)
        class_id = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # ป้องกัน index error ถ้า class ไม่ตรง
        predicted_label = CLASSES[class_id] if class_id < len(CLASSES) else "Unknown"
        
        return {"label": predicted_label, "confidence": confidence}
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"label": "Error", "confidence": 0.0}

if __name__ == "__main__":
    import uvicorn
    # เรียก load model อีกครั้งเผื่อกรณี run direct
    load_ai_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
