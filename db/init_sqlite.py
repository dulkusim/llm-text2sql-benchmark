import sqlite3
import os

# Get the absolute path to the directory where this script lives
base_dir = os.path.dirname(os.path.abspath(__file__)) 

# Create the db file in that same directory
db_path = os.path.join(base_dir, "sqlite_text2sql.db")

conn = sqlite3.connect(db_path)

# create a simple test table
conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
conn.execute("INSERT INTO test (name) VALUES ('SQLite working!')")
conn.commit()

print("✅ SQLite database created and test table inserted.")
conn.close()