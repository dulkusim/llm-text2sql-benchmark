import os
import re
import math
from decimal import Decimal
from sqlalchemy import text


# -------------------------------------------------------------------
# Schema helpers
# -------------------------------------------------------------------
def _strip_sql_comments(sql_text: str) -> str:
    """Remove /* */ and -- comments."""
    if not sql_text:
        return ""
    sql_text = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.S)
    sql_text = re.sub(r"--.*?$", "", sql_text, flags=re.M)
    return sql_text


def _semi_compact_schema(schema_text: str, max_tables: int | None = None) -> str:
    """
    Semi-compact schema for LLM prompts:
    - Keep CREATE TABLE blocks (table + column names)
    - Keep FOREIGN KEY / REFERENCES lines (to help joins)
    - Drop noisy constraints, indexes, triggers, inserts, etc.
    """
    if not schema_text:
        return ""

    schema_text = _strip_sql_comments(schema_text)
    lines = [ln.rstrip() for ln in schema_text.splitlines() if ln.strip()]

    out = []
    in_create = False
    current_table = None
    tables_kept = 0

    create_re = re.compile(r"^\s*CREATE\s+TABLE\s+`?\"?([A-Za-z0-9_]+)`?\"?\s*\(", re.IGNORECASE)

    for raw in lines:
        line = raw.strip()
        low = line.lower()

        # skip obvious noise
        if low.startswith(("insert ", "update ", "delete ", "pragma ", "begin", "commit", "drop ")):
            continue
        if low.startswith(("create index", "create trigger", "create view")):
            continue

        m = create_re.match(line)
        if m:
            # start CREATE TABLE
            if max_tables is not None and tables_kept >= max_tables:
                break

            in_create = True
            current_table = m.group(1)
            tables_kept += 1
            out.append(f"CREATE TABLE {current_table} (")
            continue

        if in_create:
            # end of CREATE TABLE block
            if line.startswith(")"):
                out.append(");")
                out.append("")  # blank line between tables
                in_create = False
                current_table = None
                continue

            # Keep FK lines (helpful for joins)
            if "foreign key" in low or "references" in low:
                # normalize spacing
                out.append(re.sub(r"\s+", " ", line).rstrip(",") + ",")
                continue

            # Skip constraints noise
            if low.startswith(("primary key", "unique", "constraint", "check")):
                continue

            # Keep only the column name (and optionally its type very lightly)
            # Typical column line: col_name TYPE ...
            col = line.split()[0].strip('`"').rstrip(",")
            if col and col.replace("_", "").isalnum():
                out.append(f"  {col},")
            continue

    # cleanup trailing commas before ");"
    cleaned = []
    for i, ln in enumerate(out):
        if ln.strip() == ");":
            # remove trailing comma from previous line if exists
            if cleaned and cleaned[-1].rstrip().endswith(","):
                cleaned[-1] = cleaned[-1].rstrip().rstrip(",")
        cleaned.append(ln)

    return "\n".join(cleaned).strip()


def get_schema_string(dataset_db_id: str, mode: str = "semi") -> str:
    """
    Returns schema DDL string for a db_id.

    Tries, in order:
      1) data/<db_id>-db.sql
      2) data/database/<db_id>/schema.sql
      3) any .sql file inside data/database/<db_id>/
      4) derive schema from a .sqlite file inside data/database/<db_id>/

    mode:
      - "full": full schema text
      - "semi": semi-compact (table+columns + FK/REFERENCES)  <-- recommended
    """
    import glob
    import sqlite3

    # --- 1) and 2) original expected paths ---
    candidates = [
        os.path.join("data", f"{dataset_db_id}-db.sql"),
        os.path.join("data", "database", dataset_db_id, "schema.sql"),
    ]

    schema_text = None
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                schema_text = f.read()
            break

    # --- 3) any .sql inside the db folder (car_1.sql, TinyCollege.sql, etc.) ---
    if schema_text is None:
        db_dir = os.path.join("data", "database", dataset_db_id)
        sql_files = sorted(glob.glob(os.path.join(db_dir, "*.sql")))
        # προτίμησε schema.sql αν υπάρχει (ακόμα κι αν δεν ήταν στα candidates),
        # αλλιώς πάρε το πρώτο .sql που βρήκες
        if sql_files:
            # αν κάπου υπάρχει schema.sql, βάλ' το πρώτο
            sql_files = sorted(sql_files, key=lambda x: (os.path.basename(x) != "schema.sql", x))
            with open(sql_files[0], "r", encoding="utf-8") as f:
                schema_text = f.read()

    # --- 4) dump schema from sqlite_master if only .sqlite exists ---
    if schema_text is None:
        db_dir = os.path.join("data", "database", dataset_db_id)
        sqlite_files = sorted(glob.glob(os.path.join(db_dir, "*.sqlite")))
        if sqlite_files:
            sqlite_path = sqlite_files[0]
            con = sqlite3.connect(sqlite_path)
            cur = con.execute("""
                SELECT sql
                FROM sqlite_master
                WHERE sql IS NOT NULL
                  AND type IN ('table','view','trigger','index')
                ORDER BY
                  CASE type
                    WHEN 'table' THEN 1
                    WHEN 'view' THEN 2
                    WHEN 'index' THEN 3
                    WHEN 'trigger' THEN 4
                    ELSE 5
                  END,
                  name;
            """)
            stmts = [r[0] for r in cur.fetchall() if r[0]]
            con.close()
            schema_text = ";\n\n".join(stmts).strip()
            if schema_text and not schema_text.endswith(";"):
                schema_text += ";"

    if schema_text is None or not schema_text.strip():
        raise FileNotFoundError(
            f"Could not derive schema for DB '{dataset_db_id}'. "
            f"Looked for *-db.sql / schema.sql / any .sql / any .sqlite."
        )

    # Apply mode
    if mode == "full":
        return schema_text
    elif mode == "semi":
        return _semi_compact_schema(schema_text)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# -------------------------------------------------------------------
# Query execution
# -------------------------------------------------------------------
def execute_query(engine, query: str):
    """
    Execute SQL on a SQLAlchemy engine.
    Returns: (success: bool, result_or_error: list[tuple] | str)

    - Always rolls back.
    - If query is an ERROR_ marker, fail immediately.
    """
    if query is None:
        return False, "Query is None"

    q = str(query).strip()

    # fast-fail if you stored generation error in the SQL string
    if q.startswith("ERROR_") or q.startswith("❌") or q.startswith("FAILED"):
        return False, q

    try:
        with engine.connect() as connection:
            trans = connection.begin()
            try:
                result_proxy = connection.execute(text(q))
                if result_proxy.returns_rows:
                    rows = result_proxy.fetchall()
                    results = [tuple(r) for r in rows]
                else:
                    results = []
                trans.rollback()
                return True, results
            except Exception as e:
                trans.rollback()
                return False, str(e)
    except Exception as e:
        return False, str(e)


# -------------------------------------------------------------------
# Result comparison (robust)
# -------------------------------------------------------------------
def _normalize_value(v):
    if v is None:
        return None

    if isinstance(v, Decimal):
        fv = float(v)
        if fv.is_integer():
            return int(fv)
        return fv

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        return v

    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="ignore").strip()
        except Exception:
            return str(v)

    return str(v).strip()


def compare_results(result_1, result_2, float_tol: float = 1e-6) -> bool:
    """
    - Ignore row order (sort rows)
    - Preserve column order
    - Tolerate float differences
    """
    if not isinstance(result_1, list) or not isinstance(result_2, list):
        return False

    def norm_rows(res):
        return [tuple(_normalize_value(x) for x in row) for row in res]

    r1 = norm_rows(result_1)
    r2 = norm_rows(result_2)

    if len(r1) == 0 and len(r2) == 0:
        return True
    if len(r1) != len(r2):
        return False

    r1s = sorted(r1, key=lambda t: tuple(str(x) for x in t))
    r2s = sorted(r2, key=lambda t: tuple(str(x) for x in t))

    for a, b in zip(r1s, r2s):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, float) or isinstance(y, float):
                try:
                    if x is None or y is None:
                        if x != y:
                            return False
                    else:
                        fx = float(x)
                        fy = float(y)
                        if math.isfinite(fx) and math.isfinite(fy):
                            if abs(fx - fy) > float_tol:
                                return False
                        else:
                            if fx != fy:
                                return False
                except Exception:
                    if str(x) != str(y):
                        return False
            else:
                if x != y:
                    return False

    return True


# -------------------------------------------------------------------
# Helper: build an error-feedback prompt for one retry
# -------------------------------------------------------------------
def build_fix_prompt(original_sql: str, error_msg: str) -> str:
    """
    Message chunk you can append to the user content for a single retry.
    """
    return (
        "The previous SQL failed to execute with the following error:\n"
        f"{error_msg}\n\n"
        "Here is the SQL that failed:\n"
        f"{original_sql}\n\n"
        "Fix the SQL using ONLY the provided schema. Output ONLY the corrected SQL query."
    )