import os
import sys
import time
import pandas as pd
import argparse
import glob
import random
from tqdm import tqdm
from sqlalchemy import create_engine, text

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from db.config import get_engine
from db.load_dataset import load_db
from utils.data_utils import load_questions
from utils.sql_utils import (
    get_schema_string,
    execute_query,
    compare_results,
    build_fix_prompt
)
from agent.model_wrappers import TinyLlamaWrapper, Qwen2Wrapper

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["tinylama", "qwen"])
parser.add_argument("--n", type=int, default=50, help="Number of questions per dataset")
parser.add_argument("--datasets", nargs="+", default=["spider", "atis", "geography", "custom"])
parser.add_argument(
    "--results_dir",
    type=str,
    default=os.path.join(project_root, "results"),
    help="Directory to store results CSV (use a Drive path for persistence)"
)
parser.add_argument(
    "--flush_every",
    type=int,
    default=25,
    help="Write progress to disk every N processed questions"
)

# Debug where it hangs
parser.add_argument("--debug_hang", action="store_true")

# Postgres query timeout (statement_timeout)
parser.add_argument("--pg_timeout_ms", type=int, default=7000)

# SQLite per-query timeout (seconds)
parser.add_argument("--sqlite_timeout_s", type=float, default=15.0)

# Optional: skip PG exec
parser.add_argument("--skip_pg_exec", action="store_true")

args = parser.parse_args()

MODELS_TO_TEST = (
    [("TinyLlama", TinyLlamaWrapper)]
    if args.model == "tinylama"
    else [("Qwen2.5-1.5B", Qwen2Wrapper)]
)

os.makedirs(args.results_dir, exist_ok=True)
RESULTS_FILE = os.path.join(args.results_dir, f"results_{args.model}.csv")

COLS = [
    "model", "dataset", "db_id", "question_id", "complexity", "question",
    "ground_truth_sql", "generated_sql", "gt_exec_success",
    "pred_exec_success_pg", "pred_exec_success_sqlite", "is_correct", "generation_time"
]

# --- Cache SQLite Paths ---
SQLITE_PATH_CACHE = {}

def find_source_sqlite(dataset_name, db_id):
    cache_key = f"{dataset_name}_{db_id}"
    if cache_key in SQLITE_PATH_CACHE:
        return SQLITE_PATH_CACHE[cache_key]

    p = os.path.join("data", f"{db_id}-db.added-in-2020.sqlite")
    if os.path.exists(p):
        SQLITE_PATH_CACHE[cache_key] = p
        return p

    db_dir = os.path.join("data", "database", db_id)
    files = sorted(glob.glob(os.path.join(db_dir, "*.sqlite")))
    if files:
        SQLITE_PATH_CACHE[cache_key] = files[0]
        return files[0]

    return None

def schema_from_sqlite(sqlite_eng) -> str:
    try:
        with sqlite_eng.connect() as conn:
            rows = conn.execute(text("""
                SELECT sql
                FROM sqlite_master
                WHERE type='table' AND sql IS NOT NULL
                ORDER BY name
            """)).fetchall()
        stmts = [r[0].strip() for r in rows if r and r[0]]
        return "\n\n".join(stmts)
    except Exception:
        return ""

def sample_questions(questions, dataset_name, N, seed=123):
    random.seed(seed)
    if len(questions) <= N:
        return questions

    if dataset_name == "spider":
        by_db = {}
        for q in questions:
            by_db.setdefault(q["db_id"], []).append(q)

        picked = []
        for db_id in sorted(by_db.keys()):
            picked.append(by_db[db_id][0])
            if len(picked) == N:
                break

        if len(picked) < N:
            picked_uids = set(str(x.get("_uid", "")) for x in picked)
            remaining = [q for q in questions if str(q.get("_uid", "")) not in picked_uids]
            picked.extend(random.sample(remaining, N - len(picked)))

        return picked

    return random.sample(questions, N)

def load_completed_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()

    try:
        df = pd.read_csv(
            csv_path,
            usecols=["dataset", "db_id", "question_id"],
            dtype=str,
            keep_default_na=False
        )
    except Exception:
        return set()

    out = set()
    for _, r in df.iterrows():
        ds = str(r.get("dataset", "")).strip()
        db = str(r.get("db_id", "")).strip()
        qid = str(r.get("question_id", "")).strip()
        if ds and db and qid and qid.lower() not in {"nan", "none"}:
            out.add((ds, db, qid))
    return out

def safe_write_results(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def set_pg_statement_timeout(pg_engine, timeout_ms: int):
    try:
        with pg_engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = {int(timeout_ms)}"))
            conn.commit()
    except Exception:
        pass

# ---------------------------
# SQLite execute with timeout
# ---------------------------
_SQLITE_EXEC_POOL = ThreadPoolExecutor(max_workers=1)

def execute_sqlite_with_timeout(sqlite_engine, sql: str, timeout_s: float):
    """
    Runs execute_query(sqlite_engine, sql) in a separate thread and returns (success, result).
    If it times out, returns (False, f"TIMEOUT after {timeout_s}s").
    """
    fut = _SQLITE_EXEC_POOL.submit(execute_query, sqlite_engine, sql)
    try:
        return fut.result(timeout=timeout_s)
    except FutureTimeoutError:
        return False, f"TIMEOUT after {timeout_s}s"
    except Exception as e:
        return False, f"ERROR: {e}"

def run_experiment_loop():
    print(f"🚀 Starting Optimization Experiment Loop for: {args.datasets}")
    print(f"💾 Results file: {RESULTS_FILE}")

    completed = load_completed_keys(RESULTS_FILE)
    if completed:
        print(f"🔁 Resume: found {len(completed)} completed (unique) keys. Will skip them.")

    try:
        pg_engine = get_engine("postgresql")
    except Exception as e:
        print(f"⚠️ Postgres Connection Error: {e}")
        return

    if not args.skip_pg_exec:
        set_pg_statement_timeout(pg_engine, args.pg_timeout_ms)

    all_results = []
    if os.path.exists(RESULTS_FILE):
        try:
            all_results = pd.read_csv(RESULTS_FILE).to_dict("records")
        except Exception:
            all_results = []

    active_sqlite_engines = {}
    schema_memory = {}
    processed_since_flush = 0

    def flush_now():
        nonlocal processed_since_flush
        safe_write_results(pd.DataFrame(all_results, columns=COLS), RESULTS_FILE)
        processed_since_flush = 0

    try:
        for model_name, ModelClass in MODELS_TO_TEST:
            print(f"\n{'='*60}\n🤖 LOADING MODEL: {model_name}\n{'='*60}")
            llm = ModelClass()

            for dataset_name in args.datasets:
                print(f"\n📦 Processing Dataset: {dataset_name.upper()}")

                questions = load_questions(dataset_name)
                if not questions:
                    print("   ⚠️ No questions loaded.")
                    continue

                print(f"   Loaded {len(questions)} questions from loader.")

                # Stable uid per loader order (needed for Spider weird IDs)
                for i, q in enumerate(questions):
                    q["_uid"] = f"{dataset_name}_{i}"

                questions = sample_questions(questions, dataset_name, args.n)
                questions.sort(key=lambda x: x.get("db_id", ""))

                keys = {(dataset_name, str(q.get("db_id", "")).strip(), q["_uid"]) for q in questions}
                print(f"   DEBUG: unique keys from questions = {len(keys)} / {len(questions)}")

                missing_questions = []
                for q in questions:
                    db_id = str(q.get("db_id", "")).strip()
                    qid = q["_uid"]
                    key = (dataset_name, db_id, qid)
                    if key not in completed:
                        missing_questions.append(q)

                print(f"   ✅ Missing to generate: {len(missing_questions)} / {len(questions)}")
                if not missing_questions:
                    continue

                unique_dbs = sorted(set(q["db_id"] for q in missing_questions if q.get("db_id")))
                print(f"   ...Pre-loading {len(unique_dbs)} databases and schemas...")

                for db_id in tqdm(unique_dbs, desc="Loading DBs"):
                    if db_id not in active_sqlite_engines:
                        src = find_source_sqlite(dataset_name, db_id)
                        if not src:
                            print(f"   ⚠️ Missing sqlite file for db_id={db_id}, skipping.")
                            continue
                        load_db(db_id, src, pg_engine)

                        # IMPORTANT: allow thread usage safely
                        active_sqlite_engines[db_id] = create_engine(
                            f"sqlite:///{src}",
                            connect_args={"timeout": 10, "check_same_thread": False}
                        )

                    if db_id not in schema_memory:
                        schema_txt = ""
                        try:
                            schema_txt = get_schema_string(db_id, mode="semi")
                        except Exception:
                            schema_txt = ""
                        if not schema_txt:
                            schema_txt = schema_from_sqlite(active_sqlite_engines[db_id])
                        schema_memory[db_id] = schema_txt

                print("   ...Starting Inference...")

                for q in tqdm(missing_questions, desc="Generating SQL"):
                    db_id = str(q.get("db_id", "")).strip()
                    qid = q["_uid"]
                    key = (dataset_name, db_id, qid)

                    if key in completed:
                        continue
                    if db_id not in active_sqlite_engines:
                        continue

                    sqlite_eng = active_sqlite_engines[db_id]
                    schema_str = schema_memory.get(db_id, "")

                    gt_sql = q.get("ground_truth_query", q.get("query", ""))
                    question_text = q.get("question", "")

                    if args.debug_hang:
                        print(f"\n[DEBUG] START uid={qid} db_id={db_id}")
                        print(f"[DEBUG] Question: {question_text[:250]}")
                        print("[DEBUG] GT sqlite exec START")

                    # 1) GT exec (sqlite) with timeout
                    gt_success, gt_result = execute_sqlite_with_timeout(
                        sqlite_eng, gt_sql, args.sqlite_timeout_s
                    )

                    if args.debug_hang:
                        print(f"[DEBUG] GT sqlite exec DONE (success={gt_success})")

                    # 2) Generation
                    start = time.time()
                    try:
                        pred_sql = llm.generate_sql(question_text, schema_str)
                    except Exception as e:
                        pred_sql = f"ERROR: {e}"
                    gen_time = time.time() - start

                    if args.debug_hang:
                        print(f"[DEBUG] GEN DONE uid={qid} in {gen_time:.2f}s")
                        print("[DEBUG] PRED sqlite exec START")

                    # 3) Pred exec (sqlite) with timeout
                    pred_success_sl, pred_result_sl = execute_sqlite_with_timeout(
                        sqlite_eng, pred_sql, args.sqlite_timeout_s
                    )

                    if args.debug_hang:
                        print(f"[DEBUG] PRED sqlite exec DONE (success={pred_success_sl})")

                    # Retry (only if sqlite failed and not model ERROR)
                    if (not pred_success_sl and isinstance(pred_result_sl, str)
                        and not str(pred_sql).startswith("ERROR")
                        and "out of memory" not in pred_result_sl.lower()
                        and "TIMEOUT" not in pred_result_sl):
                        if args.debug_hang:
                            print(f"[DEBUG] RETRY uid={qid} (sqlite err: {str(pred_result_sl)[:200]})")

                        retry_prompt = question_text + "\n\n" + build_fix_prompt(pred_sql, pred_result_sl)
                        try:
                            retry_sql = llm.generate_sql(retry_prompt, schema_str)
                            retry_suc, retry_res = execute_sqlite_with_timeout(
                                sqlite_eng, retry_sql, args.sqlite_timeout_s
                            )
                            if retry_suc:
                                pred_sql = retry_sql
                                pred_success_sl = retry_suc
                                pred_result_sl = retry_res
                        except Exception:
                            pass

                    # 4) Pred exec (postgres) — with statement_timeout
                    if args.skip_pg_exec:
                        pred_success_pg = False
                    else:
                        set_pg_statement_timeout(pg_engine, args.pg_timeout_ms)
                        if args.debug_hang:
                            print(f"[DEBUG] PRED pg exec START (timeout={args.pg_timeout_ms}ms)")
                        pred_success_pg, _ = execute_query(pg_engine, pred_sql)
                        if args.debug_hang:
                            print("[DEBUG] PRED pg exec DONE")

                    # 5) is_correct MUST remain
                    is_correct = (
                        gt_success and pred_success_sl and
                        compare_results(pred_result_sl, gt_result)
                    )

                    all_results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "db_id": db_id,
                        "question_id": qid,
                        "complexity": q.get("complexity", ""),
                        "question": question_text,
                        "ground_truth_sql": gt_sql,
                        "generated_sql": pred_sql,
                        "gt_exec_success": gt_success,
                        "pred_exec_success_pg": pred_success_pg,
                        "pred_exec_success_sqlite": pred_success_sl,
                        "is_correct": is_correct,
                        "generation_time": round(gen_time, 4),
                    })

                    completed.add(key)
                    processed_since_flush += 1

                    if processed_since_flush >= args.flush_every:
                        flush_now()

                flush_now()

    finally:
        try:
            if all_results:
                safe_write_results(pd.DataFrame(all_results, columns=COLS), RESULTS_FILE)
        except Exception:
            pass

        for eng in active_sqlite_engines.values():
            try:
                eng.dispose()
            except Exception:
                pass

    print(f"\n🏁 Finished! Results saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    run_experiment_loop()