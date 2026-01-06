from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="ThaiMed Sign AI Backend")

# ตั้งค่า CORS เพื่อให้ Frontend (React) ติดต่อได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ในโปรดักชั่นควรระบุโดเมนที่แน่นอน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

class LandmarkSequence(BaseModel):
    label: str
    sequence: List[List[float]]  # รับข้อมูล 60 เฟรม (แต่ละเฟรมคือ list ของตัวเลข)

class PredictSequence(BaseModel):
    sequence: List[List[float]]  # รับข้อมูล 60 เฟรมเพื่อทำนาย

# --- Mock Database / Logic ---

# ในที่นี้ใช้ List เก็บข้อมูลชั่วคราว (ในงานจริงควรใช้ Database หรือบันทึกเป็นไฟล์ .npy)
recorded_data = []

@app.get("/")
async def root():
    return {"message": "ThaiMed Sign AI API is running"}

@app.post("/upload_video")
async def upload_video(data: LandmarkSequence):
    """
    รับข้อมูล 60 เฟรม และชื่อท่าทาง เพื่อนำไปใช้ในการ Train โมเดล
    """
    if len(data.sequence) == 0:
        raise HTTPException(status_code=400, detail="Sequence is empty")
    
    # ตัวอย่างการเก็บข้อมูล: ในที่นี้เราเก็บไว้ใน RAM
    recorded_data.append({
        "label": data.label,
        "frames_count": len(data.sequence)
    })
    
    print(f"ได้รับข้อมูลท่าทาง: {data.label} จำนวน {len(data.sequence)} เฟรม")
    
    # TODO: บันทึกข้อมูลลงไฟล์ .csv หรือ .npy สำหรับนำไป Train
    # import numpy as np
    # np.save(f"data/{data.label}_{timestamp}.npy", np.array(data.sequence))

    return {
        "status": "success", 
        "message": f"บันทึกท่าทาง '{data.label}' เรียบร้อยแล้ว",
        "total_records": len(recorded_data)
    }

@app.post("/predict_video")
async def predict_video(data: PredictSequence):
    """
    รับข้อมูล 60 เฟรมล่าสุดจากหน้า Predict และส่งให้โมเดล LSTM/GRU วิเคราะห์
    """
    sequence_data = data.sequence
    
    # ตรวจสอบความครบถ้วนของข้อมูล
    if len(sequence_data) < 10: # อย่างน้อยต้องมีข้อมูลบ้าง
        return {"label": "กำลังเตรียมข้อมูล...", "confidence": 0}

    # --- ส่วนการทำนาย (TODO: โหลดโมเดลจริงมาใช้ที่นี่) ---
    # ตัวอย่างโครงสร้าง:
    # 1. ปรับรูปทรงข้อมูล (Reshape) ให้ตรงกับโมเดล: (1, 60, 63) หรือ (1, 60, 126)
    # 2. model.predict(input_data)
    # 3. ดึงชื่อคลาสที่มีความน่าจะเป็นสูงสุด
    
    # จำลองผลลัพธ์ (Mock Output)
    mock_labels = ["ปวดหัว", "เจ็บหน้าอก", "หายใจไม่ออก", "หมอ"]
    import random
    
    # ในหน้างานจริงจะใช้โมเดลวิเคราะห์ sequence_data
    predicted_label = "รอการเคลื่อนไหว..."
    confidence = 0.0
    
    # ตัวอย่าง: ถ้ามีข้อมูลใกล้ครบ 60 เฟรม ให้สุ่มคำออกมาโชว์
    if len(sequence_data) >= 50:
        predicted_label = random.choice(mock_labels)
        confidence = round(random.uniform(0.85, 0.99), 2)

    return {
        "label": predicted_label,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
