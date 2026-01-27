import json
import os
import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

def count_tables_in_statement(stmt):
    """
    Counts the number of tables in a parsed SQL statement to estimate join complexity.
    Handles both 'FROM t1 JOIN t2' and 'FROM t1, t2'.
    """
    table_count = 0
    in_from_clause = False

    # sqlparse flattens the tree nicely
    for token in stmt.flatten():
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            in_from_clause = True
            continue

        if token.ttype is Keyword and token.value.upper() in ('WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT', 'HAVING'):
            in_from_clause = False

        if in_from_clause:
            # If we see a comma or 'JOIN', it implies another table is coming or was just listed
            # But simpler: just count the identifiers that look like tables?
            # Actually, standard heuristic: Count explicit JOINs + count commas in FROM
            pass

    # A simpler heuristic that works for Spider/ATIS without full parsing overhead:
    # 1. Count explicit JOIN keywords
    # 2. Count commas strictly inside the FROM ... [WHERE/GROUP/etc] block

    text = str(stmt).lower()

    # Explicit JOINs
    explicit_joins = text.count(" join ")

    # Implicit Joins (tables separated by commas)
    implicit_joins = 0
    if "from" in text:
        # Extract everything between FROM and the next major keyword
        parts = re.split(r"( where | group by | order by | limit | having )", text.split("from", 1)[1])
        from_clause = parts[0]
        # Count commas in the FROM clause.
        # "FROM t1, t2" -> 1 comma = 1 join. "FROM t1, t2, t3" -> 2 commas = 2 joins.
        implicit_joins = from_clause.count(",")

    # Total effective joins is the max of either method or sum (usually distinct styles)
    # If mixed, sum is safer, but usually it's one style.
    return max(explicit_joins, implicit_joins)


def analyze_query_complexity(sql_query: str) -> str:
    """
    Classifies a SQL query into one of four Spider complexity levels (heuristic).
    """
    if not sql_query:
        return "Easy"

    # 1. Split by semicolon
    stmts = [s.strip() for s in sql_query.split(";") if s.strip()]
    if not stmts:
        return "Easy"

    # Analyze only the first complete statement
    raw_sql = stmts[0].lower()

    # 2. Calculate metrics
    # Use our new helper for joins
    join_count = 0
    # We parse it just to be safe if you want, or just use text processing
    parsed = sqlparse.parse(raw_sql)[0]
    join_count = count_tables_in_statement(parsed)

    subquery_count = raw_sql.count("select") - 1
    has_group_by = "group by" in raw_sql
    has_having = "having" in raw_sql
    has_order_by = "order by" in raw_sql
    has_set_op = any(op in raw_sql for op in [" union ", " intersect ", " except "])

    # 3. Spider Heuristics
    if (subquery_count >= 1) or (join_count > 0 and subquery_count > 0):
        return "Extra Hard"
    if (join_count >= 2 and (has_having or has_set_op or has_order_by)) or join_count >= 3:
        return "Hard"
    if (join_count >= 1 and join_count <= 2) or has_group_by:
        return "Medium"
    return "Easy"


def fix_sql_quotes(sql: str) -> str:
    """
    Fixes SQL string literals for PostgreSQL compatibility.
    """
    if not sql:
        return sql
    pattern = r'(\b(?:=|!=|<>|<|<=|>|>=|like|ilike|in)\b\s*)"(.*?)"'
    return re.sub(pattern, lambda m: f"{m.group(1)}'{m.group(2)}'", sql, flags=re.IGNORECASE)


def _pick_ground_truth_sql(query_item: dict) -> str | None:
    """
    Pick an executable ground-truth SQL.
    """
    candidates = [
        "sql-original", "sql_original", "sqlOriginal",
        "query", "query_original", "queryOriginal",
        "gold", "gold_sql", "gold_query",
        "oracle", "oracle_sql",
        "sql"
    ]
    for k in candidates:
        if k in query_item and query_item is not None:
            v = query_item.get(k)
            if isinstance(v, list) and v:
                s = v[0]
            elif isinstance(v, str):
                s = v
            else:
                continue
            if not s or not isinstance(s, str):
                continue
            s = s.strip()
            if not s:
                continue
            if re.search(r"\balias\d+\b", s, flags=re.IGNORECASE) and k != "sql":
                continue
            return s

    s = None
    if "sql" in query_item:
        v = query_item.get("sql")
        if isinstance(v, list) and v:
            s = v[0]
        elif isinstance(v, str):
            s = v
    if isinstance(s, str) and s.strip():
        return s.strip()
    return None


def load_questions(dataset_name: str = "spider"):
    """
    Loads questions from data/{dataset}.json
    """
    json_path = os.path.join("data", f"{dataset_name}.json")

    if not os.path.exists(json_path):
        print(f"❌ ERROR: Dataset file not found at {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_set = []

    for i, query_item in enumerate(data):
        gold_raw = _pick_ground_truth_sql(query_item)
        if not gold_raw:
            continue

        gold_raw = fix_sql_quotes(gold_raw)

        # Complexity is calculated on the raw gold SQL
        complexity = analyze_query_complexity(gold_raw)

        sentences = query_item.get("sentences", [])
        if not sentences:
            sentences = [{"text": query_item.get("question", ""), "variables": {}}]

        for sentence_item in sentences:
            if dataset_name == "spider":
                db_id = sentence_item.get("database")
            elif dataset_name == "custom":
                db_id = "custom"
            else:
                db_id = dataset_name

            question = sentence_item.get("text", "") or ""
            variables = sentence_item.get("variables", {}) or {}

            gold_query = gold_raw

            for var_name, var_value in variables.items():
                str_value = str(var_value)
                question = question.replace(var_name, str_value)
                gold_query = gold_query.replace(var_name, str_value)

            if re.search(r"\balias\d+\b", gold_query, flags=re.IGNORECASE) and dataset_name != "spider":
                continue

            question_set.append(
                {
                    "id": f"{dataset_name}_{i}",
                    "db_id": db_id,
                    "question": question,
                    "ground_truth_query": gold_query,
                    "complexity": complexity,
                }
            )

    print(f"Loaded {len(question_set)} questions from {json_path}.")
    return question_set


if __name__ == "__main__":
    for ds in ["spider", "atis", "geography", "custom"]:
        print(f"\n--- Loading {ds.upper()} ---")
        qs = load_questions(ds)
        if qs:
            levels = [q["complexity"] for q in qs]
            counts = {
                "Easy": levels.count("Easy"),
                "Medium": levels.count("Medium"),
                "Hard": levels.count("Hard"),
                "Extra Hard": levels.count("Extra Hard"),
            }
            print(f"Total: {len(qs)}")
            print(f"Complexity: {counts}")
            # print(f"Sample SQL: {qs[0]['ground_truth_query']}")