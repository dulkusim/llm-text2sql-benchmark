import sys
import os

# --- 1. Add project root to path ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from db.config import get_engine
    # UPDATED: Import 'load_db' instead of 'load_database'
    from db.load_dataset import load_db
    from utils.data_utils import load_questions
    from utils.sql_utils import get_schema_string, execute_query, compare_results
except ImportError as e:
    print(f"❌ ERROR: Could not import modules. Check your __init__.py files or paths.")
    print(f"Details: {e}")
    sys.exit(1)

def get_sqlite_path(db_id):
    """Helper to find the sqlite file for a given db_id."""
    # 1. Check for Repo datasets (atis, geography)
    repo_path = os.path.join("data", f"{db_id}-db.added-in-2020.sqlite")
    if os.path.exists(repo_path):
        return repo_path
    
    # 2. Check for Spider datasets (concert_singer, etc.)
    spider_path = os.path.join("data", "database", db_id, f"{db_id}.sqlite")
    if os.path.exists(spider_path):
        return spider_path
        
    return None

def run_verification():
    print("--- 🏁 Starting Phase 2 Verification (Integration Test) ---")
    
    # --- 2. Get Engines ---
    try:
        pg_engine = get_engine("postgresql")
        sqlite_engine = get_engine("sqlite")
        print("✅ Engines available (PG & SQLite).")
    except Exception as e:
        print(f"❌ FAILED to get engines: {e}")
        return

    # --- 3. Load All Questions ---
    try:
        spider_qs = load_questions("spider")
        atis_qs = load_questions("atis")
        geo_qs = load_questions("geography")
        # Note: 'academic' might be inside spider.json depending on how you loaded it, 
        # but let's stick to the main 3 for safety.
        all_qs = spider_qs + atis_qs + geo_qs
        print(f"✅ Loaded {len(all_qs)} total questions.")
    except Exception as e:
        print(f"❌ FAILED to load questions: {e}")
        return

    # --- 4. Define Test Cases ---
    test_dbs = {
        "spider": "concert_singer",
        "atis": "atis",
        "geography": "geography"
    }

    print("\n--- 🚀 Running Core Agent Loop Test ---")

    for test_name, db_id in test_dbs.items():
        print(f"\n--- Testing Dataset: {test_name.upper()} (db_id: {db_id}) ---")
        
        # --- Test 1: Load Data ---
        print(f"[Test 1/4] Finding and Loading database '{db_id}'...")
        
        # Get the path using our new helper
        sqlite_path = get_sqlite_path(db_id)
        if not sqlite_path:
            print(f"❌ FAILED: Could not find .sqlite file for '{db_id}'.")
            continue
            
        # UPDATED: Call 'load_db' with 3 arguments
        pg_load_success = load_db(db_id, sqlite_path, pg_engine)
        sqlite_load_success = load_db(db_id, sqlite_path, sqlite_engine)
        
        if not (pg_load_success and sqlite_load_success):
            print(f"❌ FAILED: Data loading for '{db_id}'.")
            continue
        print("  ✅ Data loaded into PG & SQLite.")

        # --- Test 2: Get Schema ---
        print(f"[Test 2/4] Retrieving schema string for '{db_id}'...")
        try:
            schema_string = get_schema_string(db_id)
            if "Schema not found" in schema_string or not schema_string:
                print(f"❌ FAILED: Schema string not found for '{db_id}'.")
                continue
            print(f"  ✅ Schema retrieved (length: {len(schema_string)} chars).")
        except Exception as e:
             print(f"❌ FAILED: Schema Error: {e}")
             continue

        # --- Test 3: Execute Ground Truth Query ---
        print(f"[Test 3/4] Executing sample query on '{db_id}'...")
        try:
            # Find a question for this DB
            sample_q = next(q for q in all_qs if q["db_id"] == db_id)
            query = sample_q["ground_truth_query"]
        except StopIteration:
            print(f"⚠️ WARNING: No questions found for '{db_id}' in loaded JSONs. Skipping query test.")
            continue

        print(f"  Query: {query[:70]}...")
        pg_success, pg_results = execute_query(pg_engine, query)
        sqlite_success, sqlite_results = execute_query(sqlite_engine, query)
        
        if not (pg_success and sqlite_success):
            print(f"❌ FAILED: Query execution failed.")
            print(f"  PG Error: {pg_results if not pg_success else 'OK'}")
            print(f"  SQLite Error: {sqlite_results if not sqlite_success else 'OK'}")
            continue
        print("  ✅ Query executed successfully on both DBs.")

        # --- Test 4: Compare Results ---
        print(f"[Test 4/4] Comparing results from PG and SQLite...")
        if not compare_results(pg_results, sqlite_results):
            print(f"❌ FAILED: Results are NOT identical!")
            print(f"  PG Results: {pg_results}")
            print(f"  SQLite Results: {sqlite_results}")
            continue
        print("  ✅ Results are identical.")
        print(f"--- ✅ VERIFIED: {test_name.upper()} ---")

    print("\n--- 🎉 Phase 2 Verification Complete! ---")

if __name__ == "__main__":
    run_verification()