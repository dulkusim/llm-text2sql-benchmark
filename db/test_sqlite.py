from config import get_engine
from sqlalchemy import text

def test_sqlite():
    try:
        engine = get_engine("sqlite")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM test;"))
            print("SQLite test table rows:")
            for row in result:
                print(row)
        print("✅ SQLite connection and query successful!")
    except Exception as e:
        print("❌ SQLite test failed:", e)

if __name__ == "__main__":
    test_sqlite()
