from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ⚠️ ใส่ Connection String ของคุณที่นี่
# แนะนำให้ใช้ os.getenv("DATABASE_URL") เพื่อความปลอดภัย
SQLALCHEMY_DATABASE_URL = "postgresql://thaimed_db_user:7qCAvO14szgLf3FfToANxFq5xOugRxRq@dpg-d5600t6r433s73dslnlg-a/thaimed_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency สำหรับใช้ใน FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
