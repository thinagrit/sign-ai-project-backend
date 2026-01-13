import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from database import get_db
from models import SignSequence

SEQUENCE_LENGTH = 30

def load_data_from_db():
    db = next(get_db())
    
    # 1. หา Label ทั้งหมด
    distinct_labels = db.query(SignSequence.label).distinct().all()
    actions = sorted([r[0] for r in distinct_labels])
    label_map = {label: num for num, label in enumerate(actions)}
    
    sequences = []
    labels = []
    
    print(f"Fetching data for classes: {actions}...")
    
    # 2. ดึงข้อมูลทั้งหมด
    all_data = db.query(SignSequence).all()
    
    for item in all_data:
        # data.frames ถูกเก็บเป็น JSON list อยู่แล้ว
        res = np.array(item.frames)
        
        # จัดการความยาวให้เท่ากัน (30 เฟรม)
        if len(res) == SEQUENCE_LENGTH:
            sequences.append(res)
            labels.append(label_map[item.label])
        elif len(res) > SEQUENCE_LENGTH:
            sequences.append(res[:SEQUENCE_LENGTH])
            labels.append(label_map[item.label])
            
    print(f"✅ Loaded {len(sequences)} sequences from Database")
    return np.array(sequences), to_categorical(labels).astype(int), actions

def train():
    try:
        X, y, actions = load_data_from_db()
    except Exception as e:
        print(f"Error connecting/loading DB: {e}")
        return

    if len(X) == 0:
        print("❌ Database is empty!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)),
        tf.keras.layers.LSTM(128, return_sequences=False, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    print("🚀 Start Training...")
    model.fit(X_train, y_train, epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
    
    model.save('model.h5')
    print("✅ Model saved as model.h5")

if __name__ == "__main__":
    train()
