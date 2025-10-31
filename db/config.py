import os
from sqlalchemy import create_engine

def get_engine(db_type="postgresql"):
    if db_type == "postgresql":
        # Update these with your PostgreSQL credentials
        user = "user"
        password = "pass"
        host = "localhost"
        port = "5432"
        db_name = "text2sql"
        return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}")
    
    elif db_type == "sqlite":
        # Absolute path to SQLite file, works from anywhere
        base_dir = os.path.dirname(os.path.abspath(__file__))  # folder where config.py lives
        db_path = os.path.join(base_dir, "sqlite_text2sql.db")
        return create_engine(f"sqlite:///{db_path}")
    
    else:
        raise ValueError("Unsupported DB type")