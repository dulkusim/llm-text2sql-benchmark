import json
import os
import sqlparse 

def analyze_query_complexity(sql_query):
    """
    Classifies a SQL query into one of four Spider complexity levels.
    """
    sql_lower = sql_query.lower().strip()
    
    try:
        parsed = sqlparse.parse(sql_query)
        keywords = set(str(t).lower() for t in parsed[0].tokens if t.is_keyword)
    except Exception:
        keywords = set()
    
    join_count = sql_lower.count(' join ')
    subquery_count = sql_lower.count('select') - 1
    
    has_group_by = 'group by' in sql_lower
    has_having = 'having' in sql_lower
    has_order_by = 'order by' in sql_lower
    has_set_op = any(op in sql_lower for op in ['union', 'intersect', 'except'])
    
    if (subquery_count >= 1) or (join_count > 0 and subquery_count > 0):
        return 'Extra Hard'
    if (join_count >= 2 and (has_having or has_set_op or has_order_by)) or join_count >= 3:
        return 'Hard'
    if (join_count >= 1 and join_count <= 2) or has_group_by:
        return 'Medium'
    return 'Easy'

def fix_sql_quotes(sql):
    """
    Fixes SQL string literals for PostgreSQL compatibility.
    Replaces double quotes (") with single quotes (') for values.
    """
    if not sql: return sql
    
    # This is a simple heuristic: generally in these datasets, 
    # double quotes are used for string literals.
    # We replace them with single quotes.
    fixed_sql = sql.replace('"', "'")
    
    # Sometimes the dataset might have weird spacing like = ' value '
    # but the main issue is the quote type.
    return fixed_sql

def load_questions(dataset_name="spider"):
    """
    Loads questions from the standardized JSON files in the data/ folder.
    """
    json_path = os.path.join("data", f"{dataset_name}.json")

    if not os.path.exists(json_path):
        print(f"❌ ERROR: Dataset file not found at {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    question_set = []
    
    for i, query_item in enumerate(data):
        if not query_item.get("sql"):
            continue
            
        # Fix the quotes immediately upon loading
        raw_gold_query = query_item["sql"][0]
        gold_query = fix_sql_quotes(raw_gold_query)
        
        complexity = analyze_query_complexity(gold_query)
        
        for sentence_item in query_item["sentences"]:
            if dataset_name == "spider":
                db_id = sentence_item.get("database")
            else:
                db_id = dataset_name
            
            question = sentence_item["text"]
            
            # Variable substitution
            variables = sentence_item.get("variables", {})
            for var_name, var_value in variables.items():
                # Ensure values are stringified
                str_value = str(var_value)
                question = question.replace(var_name, str_value)
                
                # For the SQL, we need to be careful.
                # If the variable was inside quotes in the original string, 
                # we just replace the name.
                # If the dataset had var_name without quotes, we might need to add them.
                # BUT, applying fix_sql_quotes FIRST usually handles the structure,
                # so simple replacement often works for these specific datasets.
                gold_query = gold_query.replace(var_name, str_value)

            question_set.append({
                "id": f"{dataset_name}_{i}",
                "db_id": db_id,
                "question": question,
                "ground_truth_query": gold_query,
                "complexity": complexity
            })
        
    print(f"Loaded {len(question_set)} questions from {json_path}.")
    return question_set

if __name__ == "__main__":
    # --- Test this script with ALL datasets ---
    datasets_to_test = ["spider", "atis", "geography"]
    
    for ds in datasets_to_test:
        print(f"\n--- Loading {ds.upper()} ---")
        qs = load_questions(ds)
        if qs:
            # Calculate stats
            levels = [q['complexity'] for q in qs]
            counts = {
                "Easy": levels.count("Easy"),
                "Medium": levels.count("Medium"),
                "Hard": levels.count("Hard"),
                "Extra Hard": levels.count("Extra Hard")
            }
            print(f"Total: {len(qs)}")
            print(f"Complexity: {counts}")
            
            # Show a sample to verify quotes are fixed (look for single quotes ')
            print(f"Sample SQL: {qs[0]['ground_truth_query']}")