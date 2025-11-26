import os
from sqlalchemy import text

def get_schema_string(dataset_db_id):
    """
    Returns the CREATE TABLE statements for a given database ID.
    It checks both the repo's format and Spider's format.
    """
    
    # Path 1: Check for repo format (e.g., 'data/atis-db.sql')
    schema_file_path = os.path.join("data", f"{dataset_db_id}-db.sql")
    
    if not os.path.exists(schema_file_path):
        # Path 2: Check for Spider format (e.g., 'data/database/concert_singer/schema.sql')
        schema_file_path = os.path.join("data", "database", dataset_db_id, "schema.sql")

    if not os.path.exists(schema_file_path):
        raise FileNotFoundError(f"Schema file not found for DB: {dataset_db_id} at {schema_file_path}")
        
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        schema = f.read()
        
    return schema.strip()

def execute_query(engine, query):
    """
    Executes a SQL query on the given engine and returns the results.
    Wraps execution in a transaction and rolls back to prevent changes.
    """
    try:
        with engine.connect() as connection:
            # Begin a transaction
            with connection.begin() as trans:
                result_proxy = connection.execute(text(query))
                
                if result_proxy.returns_rows:
                    results = [tuple(row) for row in result_proxy.fetchall()]
                    trans.rollback() # Roll back any potential changes
                    return True, results
                else:
                    # Non-row-returning query (e.g., UPDATE, INSERT, DELETE)
                    trans.rollback() # Roll back any potential changes
                    return True, [] 
                
    except Exception as e:
        # Return the error message as a string
        return False, str(e)

def compare_results(result_1, result_2):
    """
    Compares two query results (lists of tuples) for equivalence.
    Sorts them to handle different query result orderings.
    """
    if not isinstance(result_1, list) or not isinstance(result_2, list):
        return False # One was an error
        
    try:
        # Sort each tuple internally, then sort the list of tuples
        # This makes the comparison robust to column and row order
        sorted_r1 = sorted([sorted(tuple(r)) for r in result_1])
        sorted_r2 = sorted([sorted(tuple(r)) for r in result_2])
        
        return sorted_r1 == sorted_r2
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        return False