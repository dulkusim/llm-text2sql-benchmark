import os
import pandas as pd
import sqlite3
from db.config import get_engine  # Import your engine getter

def load_db(db_id, source_sqlite_file, target_engine):
    """
    Generic function to load a .sqlite file into a target engine.
    """
    if not os.path.exists(source_sqlite_file):
        print(f"❌ ERROR: Source DB file not found at {source_sqlite_file}")
        return False

    print(f"Loading database: {db_id} into {target_engine.name}...")
    
    try:
        source_conn = sqlite3.connect(source_sqlite_file)
        cursor = source_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in cursor.fetchall()]

        # Use a transaction for faster loading
        with target_engine.begin() as target_conn:
            for table_name in table_names:
                df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", source_conn)
                # 'if_exists='replace'' drops/recreates table for a clean state
                df.to_sql(table_name, target_conn, if_exists='replace', index=False)
                
        print(f"  ✅ Database '{db_id}' loaded successfully.")
        return True
    except Exception as e:
        print(f"  ❌ FAILED to load database '{db_id}': {e}")
        return False
    finally:
        if 'source_conn' in locals():
            source_conn.close()

if __name__ == "__main__":
    pg_engine = get_engine("postgresql")
    sqlite_engine = get_engine("sqlite") # Your main project SQLite DB
    
    # --- 1. Load Simple/Medium Datasets (from repo) ---
    repo_datasets = ["geography", "atis"]
    for db_id in repo_datasets:
        sqlite_file = os.path.join("data", f"{db_id}-db.added-in-2020.sqlite")
        load_db(db_id, sqlite_file, pg_engine)
        load_db(db_id, sqlite_file, sqlite_engine)

    # --- 2. Load Complex Dataset (Spider) ---
    # We will just load a few complex examples from Spider to start
    # You can add more later (e.g., "car_1", "cre_Doc_Template_Mgt")
    spider_datasets = ["concert_singer", "academic"] 
    for db_id in spider_datasets:
        # Note the different path for spider data
        sqlite_file = os.path.join("data", "database", db_id, f"{db_id}.sqlite")
        load_db(db_id, sqlite_file, pg_engine)
        load_db(db_id, sqlite_file, sqlite_engine)
        
    print("\nData loading complete.")