from config import get_engine
from sqlalchemy import text

def test_postgres():
    try:
        engine = get_engine("postgresql")
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).fetchone()
            print("PostgreSQL version:", version[0])
        print("✅ PostgreSQL connection successful!")
    except Exception as e:
        print("❌ PostgreSQL test failed:", e)

if __name__ == "__main__":
    test_postgres()
