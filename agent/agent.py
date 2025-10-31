import sys
import os
from sqlalchemy import text

# Add the project root (one level up from 'agent') to the system path
# This allows 'agent.py' to find and import modules from the 'db' directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

try:
    # Import the engine getter function from your config file
    from db.config import get_engine
except ImportError as e:
    # This handles the case if the import fails due to path issues
    print(f"❌ ERROR: Could not import db.config. Ensure your project structure is correct. ({e})")
    sys.exit(1)


def connect_and_test(db_type):
    """Initializes the engine and runs a simple query to confirm connectivity."""
    try:
        engine = get_engine(db_type)
        with engine.connect() as connection:
            
            if db_type == "postgresql":
                 # Test: Get PostgreSQL version (confirms live network connection)
                 version = connection.execute(text("SELECT version();")).fetchone()
                 print(f"✅ {db_type.capitalize()} Status: Connected (Version: {version[0][:20]}...)")
            else: # sqlite
                 # Test: Query the 'test' table you created in init_sqlite.py
                 connection.execute(text("SELECT name FROM test;")).fetchone()
                 print(f"✅ {db_type.capitalize()} Status: Connected and test table queried.")
        return engine # Return the working engine object
    except Exception as e:
        print(f"❌ {db_type.capitalize()} Connection FAILED: {e}")
        return None

def main():
    print("*** Agent Setup: Connecting Databases ***")
    
    # --- 1. Connect and test PostgreSQL ---
    # Requires Docker container to be running
    pg_engine = connect_and_test("postgresql")
    
    # --- 2. Connect and test SQLite ---
    # Requires db/sqlite_text2sql.db to have been initialized
    sqlite_engine = connect_and_test("sqlite")
    
    if pg_engine and sqlite_engine:
        print("\n*** Agent is fully ready: Both RDBMS Engines are Available ***")
        
        # Store the engines for use throughout the agent's lifetime:
        # self.pg_engine = pg_engine
        # self.sqlite_engine = sqlite_engine
        # ... (Your main logic for running LLM experiments goes here)
        
    else:
        print("\n*** Agent NOT ready: One or more database connections failed. ***")


if __name__ == "__main__":
    main()