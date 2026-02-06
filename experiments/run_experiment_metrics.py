import os
import sys
import time
import pandas as pd
import argparse
import glob
import random
from tqdm import tqdm
from sqlalchemy import create_engine, text

# ---- Per-question deps (CPU/RAM snapshots + GPU util/mem) ----
try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None
    _NVML_AVAILABLE = False

# ---- Run-level deps (peak CPU/RAM sampler + GPU peak VRAM via torch) ----
import threading
try:
    import torch
except ImportError:
    torch = None

_PROC = psutil.Process(os.getpid()) if psutil is not None else None
_CPU_CORES = (psutil.cpu_count(logical=True) if psutil is not None else None) or 1
_NVML_INITED = False


def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


# ---------------------------
# (B) Run-level sampler
# ---------------------------
class ResourceMonitor:
    """
    Lightweight background sampler for the current process.
    Tracks peak:
      - RAM RSS (bytes)
      - CPU percent (process)
    """
    def __init__(self, sample_interval_s: float = 0.5):
        self.sample_interval_s = float(sample_interval_s)
        self._stop = threading.Event()
        self._thread = None

        self.peak_rss_bytes = 0
        self.peak_cpu_percent = 0.0

        self._proc = None
        self._enabled = psutil is not None

    def start(self):
        if not self._enabled:
            return
        self._proc = psutil.Process(os.getpid())

        # Prime cpu_percent so subsequent reads are meaningful
        try:
            self._proc.cpu_percent(interval=None)
        except Exception:
            pass

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:
                pass

            try:
                cpu_p = self._proc.cpu_percent(interval=None)
                if cpu_p > self.peak_cpu_percent:
                    self.peak_cpu_percent = cpu_p
            except Exception:
                pass

            self._stop.wait(self.sample_interval_s)

    def stop(self):
        if not self._enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        if not self._enabled:
            return {"cpu_ram_available": False}
        return {
            "cpu_ram_available": True,
            "peak_ram_rss_mb": round(_bytes_to_mb(self.peak_rss_bytes), 2),
            "peak_cpu_percent_process": round(self.peak_cpu_percent, 1),
        }


def gpu_reset_peaks():
    if torch is None:
        return
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_summary_runlevel() -> dict:
    if torch is None:
        return {"gpu_available": False}
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return {"gpu_available": False}
    try:
        max_alloc = torch.cuda.max_memory_allocated()
        max_resv = torch.cuda.max_memory_reserved() if hasattr(torch.cuda, "max_memory_reserved") else 0
        return {
            "gpu_available": True,
            "peak_gpu_vram_allocated_mb": round(_bytes_to_mb(max_alloc), 2),
            "peak_gpu_vram_reserved_mb": round(_bytes_to_mb(max_resv), 2),
        }
    except Exception:
        return {"gpu_available": False}


# ---------------------------
# (A) Per-question metrics
# ---------------------------
def _nvml_init_once() -> bool:
    global _NVML_INITED
    if not _NVML_AVAILABLE:
        return False
    if _NVML_INITED:
        return True
    try:
        pynvml.nvmlInit()
        _NVML_INITED = True
        return True
    except Exception:
        return False


def _read_gpu_metrics_per_question():
    """
    Returns (gpu_util_percent, mem_used_mb, mem_total_mb, mem_percent) or (None,...)
    Uses GPU index 0.
    """
    if not _nvml_init_once():
        return (None, None, None, None)
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # %
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem.used / (1024 * 1024)
        total_mb = mem.total / (1024 * 1024)
        mem_percent = (used_mb / total_mb * 100.0) if total_mb else None
        return (float(util), float(used_mb), float(total_mb), float(mem_percent))
    except Exception:
        return (None, None, None, None)


def metrics_start():
    """
    Baseline for per-question CPU% computation.
    """
    if _PROC is None:
        return {"t": time.perf_counter(), "cpu_times": None}
    return {"t": time.perf_counter(), "cpu_times": _PROC.cpu_times()}


def metrics_end(start_snapshot):
    """
    Per-question metrics for whole question pipeline.
    CPU% computed from CPU time delta / wall time / cores.
    """
    t1 = time.perf_counter()
    wall = max(t1 - start_snapshot["t"], 1e-9)

    cpu_proc_percent = None
    cpu_system_percent = None
    ram_proc_rss_mb = None
    ram_system_percent = None

    if _PROC is not None and start_snapshot.get("cpu_times") is not None:
        ct0 = start_snapshot["cpu_times"]
        ct1 = _PROC.cpu_times()
        cpu_time_delta = (ct1.user - ct0.user) + (ct1.system - ct0.system)
        cpu_proc_percent = (cpu_time_delta / (wall * _CPU_CORES)) * 100.0

        try:
            ram_proc_rss_mb = _PROC.memory_info().rss / (1024 * 1024)
        except Exception:
            ram_proc_rss_mb = None

        try:
            cpu_system_percent = psutil.cpu_percent(interval=None) if psutil is not None else None
        except Exception:
            cpu_system_percent = None

        try:
            ram_system_percent = psutil.virtual_memory().percent if psutil is not None else None
        except Exception:
            ram_system_percent = None

    gpu_util, gpu_used, gpu_total, gpu_mem_percent = _read_gpu_metrics_per_question()

    def _r(x):
        return None if x is None else round(float(x), 4)

    return {
        "cpu_proc_percent": _r(cpu_proc_percent),
        "cpu_system_percent": _r(cpu_system_percent),
        "ram_proc_rss_mb": _r(ram_proc_rss_mb),
        "ram_system_percent": _r(ram_system_percent),
        "gpu_util_percent": _r(gpu_util),
        "gpu_mem_used_mb": _r(gpu_used),
        "gpu_mem_total_mb": _r(gpu_total),
        "gpu_mem_percent": _r(gpu_mem_percent),
    }


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
    build_fix_prompt,
    normalize_sql_literals,
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

# SQLite / Postgres execution budgets
parser.add_argument("--gt_timeout", type=float, default=8.0, help="Timeout seconds for GT execution (SQLite). (fast-fail)")
parser.add_argument("--pred_timeout", type=float, default=20.0, help="Timeout seconds for Pred execution (SQLite).")
parser.add_argument("--gt_row_cap", type=int, default=20000, help="Max rows to fetch for GT (SQLite).")
parser.add_argument("--pred_row_cap", type=int, default=2000, help="Max rows to fetch for Pred (SQLite).")

# Postgres timeouts
parser.add_argument("--pg_timeout_ms", type=int, default=20000, help="Postgres statement_timeout (ms) for Pred (default).")
parser.add_argument("--gt_pg_timeout_ms", type=int, default=60000, help="Postgres statement_timeout (ms) for GT fallback.")
parser.add_argument("--pred_pg_timeout_ms", type=int, default=30000, help="Postgres statement_timeout (ms) for Pred when comparing on PG.")

# Run-level sampling interval (seconds)
parser.add_argument("--resource_sample_s", type=float, default=0.5, help="Run-level resource sampling interval (s).")

args = parser.parse_args()


MODELS_TO_TEST = (
    [("TinyLlama", TinyLlamaWrapper)]
    if args.model == "tinylama"
    else [("Qwen2.5-1.5B", Qwen2Wrapper)]
)

os.makedirs(args.results_dir, exist_ok=True)
RESULTS_FILE = os.path.join(args.results_dir, f"results_{args.model}.csv")

# NOTE: This matches your current CSV schema (per-question + run-level)
COLS = [
    "model", "dataset", "db_id", "question_id", "complexity", "question",
    "ground_truth_sql", "generated_sql",
    "gt_exec_success", "pred_exec_success_pg",
    "pred_exec_success_sqlite", "is_correct", "generation_time",

    # per-question
    "cpu_proc_percent", "cpu_system_percent",
    "ram_proc_rss_mb", "ram_system_percent",
    "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_mem_percent",

    # run-level (backfilled)
    "total_runtime_s",
    "peak_ram_rss_mb",
    "peak_cpu_percent_process",
    "peak_gpu_vram_allocated_mb",
    "peak_gpu_vram_reserved_mb",
    "cpu_ram_available",
    "gpu_available",
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


def run_gt_with_fallback(sqlite_eng, pg_engine, gt_sql: str, sqlite_timeout_s: float, pg_timeout_ms: int,
                         sqlite_row_cap: int):
    """
    Try GT on SQLite first (fast-fail). If it fails due to timeout-like reasons,
    fallback to Postgres with higher statement_timeout.

    Returns:
      (gt_success, gt_result, gt_engine_used, gt_error)
      gt_engine_used in {"sqlite", "pg"}
    """
    s_success, s_res = execute_query(
        sqlite_eng,
        gt_sql,
        timeout_s=sqlite_timeout_s,
        row_cap=sqlite_row_cap,
        truncate_is_error=False
    )
    if s_success:
        return True, s_res, "sqlite", ""

    err = str(s_res)
    heavy_signals = ["interrupted", "timeout"]
    do_fallback = any(sig in err.lower() for sig in heavy_signals)

    if not do_fallback:
        return False, s_res, "sqlite", err

    p_success, p_res = execute_query(
        pg_engine,
        gt_sql,
        pg_timeout_ms=pg_timeout_ms,
        row_cap=sqlite_row_cap,
        truncate_is_error=False
    )
    if p_success:
        return True, p_res, "pg", ""
    return False, p_res, "pg", str(p_res)


def run_experiment_loop():
    # ---------------------------
    # (B) Start run-level monitoring
    # ---------------------------
    monitor = ResourceMonitor(sample_interval_s=args.resource_sample_s)
    gpu_reset_peaks()
    t0_run = time.time()
    monitor.start()

    print(f"🚀 Starting Optimization Experiment Loop for: {args.datasets}")
    print(f"💾 Results file: {RESULTS_FILE}")

    completed = load_completed_keys(RESULTS_FILE)
    if completed:
        print(f"🔁 Resume: found {len(completed)} completed (unique) keys. Will skip them.")

    try:
        pg_engine = get_engine("postgresql")
    except Exception as e:
        print(f"⚠️ Postgres Connection Error: {e}")
        monitor.stop()
        return

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

                for i, q in enumerate(questions):
                    q["_uid"] = f"{dataset_name}_{i}"

                questions = sample_questions(questions, dataset_name, args.n)
                questions.sort(key=lambda x: x.get("db_id", ""))

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
                        active_sqlite_engines[db_id] = create_engine(f"sqlite:///{src}")

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
                    # ---------------------------
                    # (A) Start per-question metrics window
                    # ---------------------------
                    m0 = metrics_start()

                    db_id = str(q.get("db_id", "")).strip()
                    qid = q["_uid"]
                    key = (dataset_name, db_id, qid)

                    if key in completed:
                        continue
                    if db_id not in active_sqlite_engines:
                        continue

                    sqlite_eng = active_sqlite_engines[db_id]
                    schema_str = schema_memory.get(db_id, "")

                    gt_sql_raw = q.get("ground_truth_query", q.get("query", ""))
                    gt_sql = normalize_sql_literals(gt_sql_raw)
                    question_text = q.get("question", "")

                    gt_success, gt_result, gt_engine_used, gt_error = run_gt_with_fallback(
                        sqlite_eng,
                        pg_engine,
                        gt_sql,
                        sqlite_timeout_s=args.gt_timeout,
                        pg_timeout_ms=args.gt_pg_timeout_ms,
                        sqlite_row_cap=args.gt_row_cap
                    )

                    start = time.time()
                    try:
                        pred_sql_raw = llm.generate_sql(question_text, schema_str)
                        pred_sql = normalize_sql_literals(pred_sql_raw)
                    except Exception as e:
                        pred_sql = f"ERROR: {e}"
                    gen_time = time.time() - start

                    pred_success_sl, pred_result_sl = execute_query(
                        sqlite_eng,
                        pred_sql,
                        timeout_s=args.pred_timeout,
                        row_cap=args.pred_row_cap,
                        truncate_is_error=True
                    )

                    if (
                        not pred_success_sl
                        and isinstance(pred_result_sl, str)
                        and not str(pred_sql).startswith("ERROR")
                        and "out of memory" not in pred_result_sl.lower()
                        and "interrupted" not in pred_result_sl.lower()
                        and "timeout" not in pred_result_sl.lower()
                    ):
                        retry_prompt = question_text + "\n\n" + build_fix_prompt(pred_sql, pred_result_sl)
                        try:
                            retry_sql_raw = llm.generate_sql(retry_prompt, schema_str)
                            retry_sql = normalize_sql_literals(retry_sql_raw)
                            retry_suc, retry_res = execute_query(
                                sqlite_eng,
                                retry_sql,
                                timeout_s=args.pred_timeout,
                                row_cap=args.pred_row_cap,
                                truncate_is_error=True
                            )
                            if retry_suc:
                                pred_sql = retry_sql
                                pred_success_sl = retry_suc
                                pred_result_sl = retry_res
                        except Exception:
                            pass

                    need_pg_results_for_compare = (gt_engine_used == "pg")

                    pg_timeout = args.pred_pg_timeout_ms if need_pg_results_for_compare else args.pg_timeout_ms
                    pg_row_cap = args.gt_row_cap if need_pg_results_for_compare else args.pred_row_cap
                    pg_trunc_err = False if need_pg_results_for_compare else True

                    pred_success_pg, pred_res_pg = execute_query(
                        pg_engine,
                        pred_sql,
                        pg_timeout_ms=pg_timeout,
                        row_cap=pg_row_cap,
                        truncate_is_error=pg_trunc_err
                    )

                    if gt_engine_used == "sqlite":
                        is_correct = (
                            gt_success and pred_success_sl and
                            compare_results(pred_result_sl, gt_result)
                        )
                    else:
                        is_correct = (
                            gt_success and pred_success_pg and
                            compare_results(pred_res_pg, gt_result)
                        )

                    # ---------------------------
                    # (A) End per-question metrics window
                    # ---------------------------
                    qm = metrics_end(m0)

                    # placeholders for run-level (backfilled later in finally)
                    run_placeholders = {
                        "total_runtime_s": None,
                        "peak_ram_rss_mb": None,
                        "peak_cpu_percent_process": None,
                        "peak_gpu_vram_allocated_mb": None,
                        "peak_gpu_vram_reserved_mb": None,
                        "cpu_ram_available": None,
                        "gpu_available": None,
                    }

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

                        # per-question metrics
                        **qm,

                        # run-level placeholders
                        **run_placeholders,
                    })

                    completed.add(key)
                    processed_since_flush += 1

                    if processed_since_flush >= args.flush_every:
                        flush_now()

                flush_now()

    finally:
        # ---------------------------
        # (B) Stop run-level monitoring and backfill run-level metrics
        # ---------------------------
        monitor.stop()
        total_runtime_s = round(time.time() - t0_run, 4)

        cpu_ram = monitor.summary()
        gpu_run = gpu_summary_runlevel()

        run_metrics = {
            "total_runtime_s": total_runtime_s,
            "peak_ram_rss_mb": cpu_ram.get("peak_ram_rss_mb", None),
            "peak_cpu_percent_process": cpu_ram.get("peak_cpu_percent_process", None),
            "cpu_ram_available": bool(cpu_ram.get("cpu_ram_available", False)),
            "peak_gpu_vram_allocated_mb": gpu_run.get("peak_gpu_vram_allocated_mb", None),
            "peak_gpu_vram_reserved_mb": gpu_run.get("peak_gpu_vram_reserved_mb", None),
            "gpu_available": bool(gpu_run.get("gpu_available", False)),
        }

        if all_results:
            for r in all_results:
                r.update(run_metrics)

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

        if _NVML_INITED and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    print(f"\n🏁 Finished! Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment_loop()
