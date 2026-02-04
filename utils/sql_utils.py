import os
import re
import math
import time
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
            if max_tables is not None and tables_kept >= max_tables:
                break

            in_create = True
            current_table = m.group(1)
            tables_kept += 1
            out.append(f"CREATE TABLE {current_table} (")
            continue

        if in_create:
            if line.startswith(")"):
                out.append(");")
                out.append("")
                in_create = False
                current_table = None
                continue

            if "foreign key" in low or "references" in low:
                out.append(re.sub(r"\s+", " ", line).rstrip(",") + ",")
                continue

            if low.startswith(("primary key", "unique", "constraint", "check")):
                continue

            col = line.split()[0].strip('`"').rstrip(",")
            if col and col.replace("_", "").isalnum():
                out.append(f"  {col},")
            continue

    cleaned = []
    for ln in out:
        if ln.strip() == ");":
            if cleaned and cleaned[-1].rstrip().endswith(","):
                cleaned[-1] = cleaned[-1].rstrip().rstrip(",")
        cleaned.append(ln)

    return "\n".join(cleaned).strip()


def get_schema_string(dataset_db_id: str, mode: str = "semi") -> str:
    import glob
    import sqlite3

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

    if schema_text is None:
        db_dir = os.path.join("data", "database", dataset_db_id)
        sql_files = sorted(glob.glob(os.path.join(db_dir, "*.sql")))
        if sql_files:
            sql_files = sorted(sql_files, key=lambda x: (os.path.basename(x) != "schema.sql", x))
            with open(sql_files[0], "r", encoding="utf-8") as f:
                schema_text = f.read()

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

    if mode == "full":
        return schema_text
    elif mode == "semi":
        return _semi_compact_schema(schema_text)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# -------------------------------------------------------------------
# SQL normalization (ATIS fix)
# -------------------------------------------------------------------
_double_quote_literal = re.compile(r'"([^"]*)"')

def normalize_sql_literals(sql: str) -> str:
    """
    Convert "TEXT" -> 'TEXT'. Escapes single quotes inside TEXT.
    Assumption: datasets like ATIS rarely rely on quoted identifiers.
    """
    if sql is None:
        return sql
    s = str(sql)
    return _double_quote_literal.sub(lambda m: "'" + m.group(1).replace("'", "''") + "'", s)


# -------------------------------------------------------------------
# Query execution
# -------------------------------------------------------------------
def execute_query(
    engine,
    query: str,
    *,
    timeout_s: float = 25.0,
    pg_timeout_ms: int = 15000,
    row_cap: int = 2000,
    truncate_is_error: bool = True
):
    """
    Execute SQL on a SQLAlchemy engine.
    Returns: (success: bool, result_or_error: list[tuple] | str)

    - Normalizes ATIS-style string literals: "TEXT" -> 'TEXT'
    - For SQLite: uses progress handler to enforce wall-clock timeout
    - For Postgres: sets LOCAL statement_timeout
    - Always rolls back.
    - Caps result size to row_cap (fetchmany(row_cap + 1)).
    - If truncate_is_error=True and result exceeds row_cap -> returns False.
      If truncate_is_error=False and result exceeds row_cap -> returns True (but truncated result returned).
    """
    if query is None:
        return False, "Query is None"

    q = normalize_sql_literals(str(query)).strip()

    low = q.lower()
    if q.startswith("ERROR_") or q.startswith("❌") or q.startswith("FAILED") or low.startswith("error:"):
        return False, q

    try:
        dialect = getattr(engine, "dialect", None)
        dialect_name = getattr(dialect, "name", "") if dialect else ""
    except Exception:
        dialect_name = ""

    try:
        with engine.connect() as connection:
            trans = connection.begin()
            raw_sqlite = None
            start_time = None

            try:
                if dialect_name == "postgresql":
                    try:
                        connection.execute(text(f"SET LOCAL statement_timeout = {int(pg_timeout_ms)};"))
                    except Exception:
                        pass

                if dialect_name == "sqlite":
                    try:
                        start_time = time.time()
                        raw_sqlite = connection.connection

                        def progress_handler():
                            return 1 if (time.time() - start_time) > float(timeout_s) else 0

                        raw_sqlite.set_progress_handler(progress_handler, 100000)
                    except Exception:
                        raw_sqlite = None

                result_proxy = connection.execute(text(q))

                if result_proxy.returns_rows:
                    rows = result_proxy.fetchmany(int(row_cap) + 1)
                    truncated = len(rows) > int(row_cap)
                    rows = rows[:int(row_cap)]
                    results = [tuple(r) for r in rows]

                    trans.rollback()

                    if truncated and truncate_is_error:
                        return False, f"Result too large (truncated at {row_cap} rows)"
                    return True, results

                trans.rollback()
                return True, []

            except Exception as e:
                trans.rollback()
                return False, str(e)

            finally:
                if raw_sqlite is not None:
                    try:
                        raw_sqlite.set_progress_handler(None, 0)
                    except Exception:
                        pass

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
    return (
        "The previous SQL failed to execute with the following error:\n"
        f"{error_msg}\n\n"
        "Here is the SQL that failed:\n"
        f"{original_sql}\n\n"
        "Fix the SQL using ONLY the provided schema. Output ONLY the corrected SQL query."
    )
