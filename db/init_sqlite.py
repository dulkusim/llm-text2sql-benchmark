import sqlite3
import os

# ensure the db folder exists
os.makedirs("db", exist_ok=True)

# connect (this automatically creates the file if it doesn’t exist)
conn = sqlite3.connect("./sqlite_text2sql.db")

# create a simple test table
conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
conn.execute("INSERT INTO test (name) VALUES ('SQLite working!')")
conn.commit()

print("✅ SQLite database created and test table inserted.")
conn.close()