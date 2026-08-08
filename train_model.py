"""
train_model.py
================
เทรนโมเดล Deep Learning (PyTorch MLP) สำหรับแยกแยะท่ามือ
จากข้อมูลที่ export ออกมาจากระบบ ThaiMed AI (/export endpoint)

วิธีใช้:
  1. ไปที่หน้า "คลังคำศัพท์" ในเว็บ กด "Export JSON" -> จะได้ไฟล์ signai-dataset-YYYY-MM-DD.json
  2. ติดตั้ง dependency:
       pip install torch scikit-learn numpy
  3. รัน:
       python train_model.py --data signai-dataset-2026-08-08.json --out sign_model.pt

ผลลัพธ์ที่ได้:
  - sign_model.pt      : weight ของโมเดล (state_dict)
  - label_encoder.json : mapping ระหว่าง index <-> ชื่อ label (เช่น "ปวดหัว_1")
  - model_meta.json     : ขนาด input/output ของโมเดล เอาไว้โหลดตอน inference
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ต้องตรงกับ backend (main.py): hand(63) + pose(27) = 90 raw จุด
# feature vector เต็มรูปแบบ (raw+velocity+magnitude+wristVx+wristVy) มีได้สองขนาด:
#   แบบใหม่ (มี pose): 90*2 + 3 = 183
#   แบบเก่า (มือเท่านั้น): 63*2 + 3 = 129
TARGET_DIM = 183  # เราจะ pad ทุกตัวอย่างให้มีขนาดนี้เสมอ เพื่อให้ input สม่ำเสมอ


def pad_or_truncate(vec, target_len=TARGET_DIM):
    """ทำให้ feature vector ทุกตัวมีความยาวเท่ากัน (pad ด้วย 0 ถ้าสั้นกว่า)"""
    vec = list(vec)
    if len(vec) >= target_len:
        return vec[:target_len]
    return vec + [0.0] * (target_len - len(vec))


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat = data.get("flat", [])
    if not flat:
        raise ValueError("ไม่พบข้อมูลในไฟล์ (key 'flat' ว่างเปล่า) — export ไฟล์ใหม่จากเว็บ")

    X, y = [], []
    for item in flat:
        points = item.get("points")
        label = item.get("label")
        if not points or not label:
            continue
        X.append(pad_or_truncate(points))
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SignMLP(nn.Module):
    """MLP classifier: รับ feature vector 1 เฟรม -> ทำนาย label"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train(args):
    print(f"กำลังโหลดข้อมูลจาก {args.data} ...")
    X, y_raw = load_dataset(args.data)
    print(f"จำนวนตัวอย่างทั้งหมด: {len(X)} | ขนาด feature: {X.shape[1]}")

    # เช็คว่าแต่ละ label มีตัวอย่างพอสมควร (เตือนถ้าน้อยเกินไป)
    unique, counts = np.unique(y_raw, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        if cnt < 10:
            print(f"  ⚠️  label '{lbl}' มีแค่ {cnt} ตัวอย่าง — แนะนำอย่างน้อย 20+")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"จำนวนท่าทั้งหมด (classes): {num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if min(counts) >= 2 else None
    )

    train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(SignDataset(X_val, y_val), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignMLP(input_dim=X.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total if total else 0.0
        scheduler.step(val_acc)

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs} | loss={total_loss/len(X_train):.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)

    print(f"\n✅ เทรนเสร็จ — val accuracy ดีที่สุด: {best_val_acc:.3f}")
    print(f"บันทึกโมเดลไว้ที่: {args.out}")

    with open("label_encoder.json", "w", encoding="utf-8") as f:
        json.dump({"classes": le.classes_.tolist()}, f, ensure_ascii=False, indent=2)

    with open("model_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_dim": int(X.shape[1]),
            "num_classes": int(num_classes),
            "target_dim": TARGET_DIM,
        }, f, indent=2)

    print("บันทึก label_encoder.json และ model_meta.json แล้ว")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path ไฟล์ JSON ที่ export จากเว็บ")
    parser.add_argument("--out", default="sign_model.pt", help="path ไฟล์โมเดลที่จะบันทึก")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args)
