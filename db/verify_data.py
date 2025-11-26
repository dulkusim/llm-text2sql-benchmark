import os
from db.config import get_engine
from sqlalchemy import text

def run_test_query(engine, query, label=""):
    """Helper to run a query and print the results."""
    print(f"--- [{label} on {engine.name.upper()}] ---")
    print(f"Query: {query}")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            print(f"Result: {result}")
            return result
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None

def verify_data_loading():
    print("🚀 Starting Database Verification...")

    # 1. Get database engines
    try:
        pg_engine = get_engine("postgresql")
        sqlite_engine = get_engine("sqlite")
    except Exception as e:
        print(f"❌ Critical Error: Could not create engines: {e}")
        return

    # --- 2. Define Test Queries ---
    # We will check one table from each dataset you loaded
    
    # Test 1: geography (Simple)
    query_geo_count = "SELECT COUNT(*) FROM city;"
    query_geo_sample = "SELECT * FROM city LIMIT 3;"
    
    # Test 2: atis (Medium)
    query_atis_count = "SELECT COUNT(*) FROM flight;"
    query_atis_sample = "SELECT * FROM flight LIMIT 3;"
    
    # Test 3: concert_singer (Spider - Complex)
    query_spider_count = "SELECT COUNT(*) FROM singer;"
    query_spider_sample = "SELECT * FROM singer LIMIT 3;"

    # --- 3. Run Tests ---
    print("\n" + "="*20 + " GEOGRAPHY " + "="*20)
    pg_geo_count = run_test_query(pg_engine, query_geo_count, "Geo Count")
    sql_geo_count = run_test_query(sqlite_engine, query_geo_count, "Geo Count")
    run_test_query(pg_engine, query_geo_sample, "Geo Sample")
    run_test_query(sqlite_engine, query_geo_sample, "Geo Sample")

    print("\n" + "="*20 + " ATIS " + "="*20)
    pg_atis_count = run_test_query(pg_engine, query_atis_count, "ATIS Count")
    sql_atis_count = run_test_query(sqlite_engine, query_atis_count, "ATIS Count")
    run_test_query(pg_engine, query_atis_sample, "ATIS Sample")
    run_test_query(sqlite_engine, query_atis_sample, "ATIS Sample")

    print("\n" + "="*20 + " SPIDER (concert_singer) " + "="*20)
    pg_spider_count = run_test_query(pg_engine, query_spider_count, "Spider Count")
    sql_spider_count = run_test_query(sqlite_engine, query_spider_count, "Spider Count")
    run_test_query(pg_engine, query_spider_sample, "Spider Sample")
    run_test_query(sqlite_engine, query_spider_sample, "Spider Sample")

    # --- 4. Final Verification ---
    print("\n" + "="*20 + " FINAL VERDICT " + "="*20)
    
    # Check if counts are valid (not None and > 0)
    counts_valid = all([
        pg_geo_count and sql_geo_count and pg_geo_count[0][0] > 0,
        pg_atis_count and sql_atis_count and pg_atis_count[0][0] > 0,
        pg_spider_count and sql_spider_count and pg_spider_count[0][0] > 0
    ])
    
    # Check if counts match between DBs
    counts_match = all([
        pg_geo_count == sql_geo_count,
        pg_atis_count == sql_atis_count,
        pg_spider_count == sql_spider_count
    ])

    if counts_valid and counts_match:
        print("✅ SUCCESS: All test tables are populated and data counts match between PostgreSQL and SQLite.")
    else:
        print("❌ FAILED: Data verification failed.")
        if not counts_valid:
            print("  - Reason: At least one table is empty or failed to query.")
        if not counts_match:
            print("  - Reason: Data counts do not match between databases.")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Make sure you have already run these two commands in your terminal:
    # 1. docker compose -f db/docker-compose.yml up -d
    # 2. python db/load_dataset.py
    #
    # If you have, you can run this script.
    verify_data_loading()