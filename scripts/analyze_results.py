import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Display settings (avoid "...")
# ---------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
results_dir = os.path.join(project_root, "results")

# ---------------------------
# Consistent colors across all plots
# ---------------------------
PALETTE_NAME = "viridis"

def get_model_palette(df):
    models = sorted(df["model"].dropna().unique().tolist())
    colors = sns.color_palette(PALETTE_NAME, n_colors=len(models))
    return {m: c for m, c in zip(models, colors)}

def load_data():
    """Loads and merges all result CSVs found in the results folder."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

    if not csv_files:
        print(f"❌ No result CSV files found in {results_dir}")
        print("   Run 'python experiments/run_experiment.py --model [tinylama|qwen]' first.")
        sys.exit(1)

    df_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            temp_df["__source_file"] = os.path.basename(f)
            df_list.append(temp_df)
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")

    if not df_list:
        sys.exit(1)

    full_df = pd.concat(df_list, ignore_index=True)

    # Enforce Complexity Order (if exists)
    if "complexity" in full_df.columns:
        complexity_order = ["Easy", "Medium", "Hard", "Extra Hard"]
        full_df["complexity"] = pd.Categorical(
            full_df["complexity"],
            categories=complexity_order,
            ordered=True
        )

    return full_df

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def analyze_difficulty_distribution(df):
    print_header("1. COMPLEXITY DISTRIBUTION (Count of Questions)")

    if "complexity" not in df.columns:
        print("⚠️ Column 'complexity' not found.")
        return

    dist = df.groupby(["dataset", "complexity"], observed=False).size().unstack(fill_value=0)
    dist["Total"] = dist.sum(axis=1)

    print(dist.to_string())
    print("\nObservation: This shows how balanced your test set was per dataset.")

def analyze_performance(df):
    print_header("2. MODEL PERFORMANCE (Accuracy & Latency)")

    metrics = df.groupby(["model", "dataset"]).agg({
        "is_correct": "mean",
        "pred_exec_success_sqlite": "mean",
        "pred_exec_success_pg": "mean",
        "generation_time": ["mean", "median"],
        "question_id": "count"
    })

    # flatten columns
    metrics.columns = ["_".join(c).strip("_") for c in metrics.columns.to_flat_index()]
    metrics = metrics.rename(columns={"question_id_count": "samples"})

    metrics["Accuracy %"] = (metrics["is_correct_mean"] * 100).round(2)
    metrics["SQLite Exec %"] = (metrics["pred_exec_success_sqlite_mean"] * 100).round(2)
    metrics["Postgres Exec %"] = (metrics["pred_exec_success_pg_mean"] * 100).round(2)
    metrics["Avg Time (s)"] = metrics["generation_time_mean"].round(4)
    metrics["Median Time (s)"] = metrics["generation_time_median"].round(4)

    display_cols = ["samples", "Accuracy %", "SQLite Exec %", "Postgres Exec %", "Avg Time (s)", "Median Time (s)"]
    print(metrics[display_cols].to_string())

def analyze_dialect_robustness(df):
    print_header("3. DIALECT ROBUSTNESS (Postgres vs SQLite)")

    if "pred_exec_success_sqlite" not in df.columns or "pred_exec_success_pg" not in df.columns:
        print("⚠️ Missing exec columns to compare dialect robustness.")
        return

    sqlite_valid = df[df["pred_exec_success_sqlite"] == True].copy()

    if sqlite_valid.empty:
        print("No queries executed successfully in SQLite to compare.")
        return

    robustness = sqlite_valid.groupby("model").agg({
        "pred_exec_success_pg": "mean"
    })

    robustness["PG Compatibility %"] = (robustness["pred_exec_success_pg"] * 100).round(2)

    print("Of queries that worked in SQLite, what % also worked in Postgres?")
    print(robustness[["PG Compatibility %"]].to_string())
    print("\nNote: Lower percentages indicate the model is relying on SQLite-specific loose typing or syntax.")

def head_to_head_comparison(df):
    print_header("4. HEAD-TO-HEAD SUMMARY")

    summary = df.groupby("model").agg({
        "is_correct": "mean",
        "generation_time": "mean",
        "pred_exec_success_sqlite": "mean"
    })

    print(f"{'Model':<20} | {'Accuracy':<10} | {'Exec Rate':<10} | {'Time/Query':<10}")
    print("-" * 60)

    for model, row in summary.iterrows():
        acc = f"{row['is_correct']*100:.2f}%"
        exe = f"{row['pred_exec_success_sqlite']*100:.2f}%"
        t = f"{row['generation_time']:.4f}s"
        print(f"{model:<20} | {acc:<10} | {exe:<10} | {t:<10}")

# ---------------------------
# NEW: Resource metrics analysis for the NEW results format
# ---------------------------
RESOURCE_COLS_NEW = [
    "cpu_proc_percent",
    "cpu_system_percent",
    "ram_proc_rss_mb",
    "ram_system_percent",
    "gpu_util_percent",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_mem_percent",
]

def _resource_subset_new(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that contain the new per-question resource metrics.
    Works even if results_dir has mixed CSVs (older ones without these cols).
    """
    needed = ["model", "generation_time"] + RESOURCE_COLS_NEW
    if not set(needed).issubset(set(df.columns)):
        return df.iloc[0:0].copy()

    sub = df.copy()
    for c in RESOURCE_COLS_NEW + ["generation_time"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # keep rows that actually have metrics (cpu_proc_percent is the best indicator)
    sub = sub[sub["cpu_proc_percent"].notna()]
    return sub

def analyze_resource_metrics(df):
    print_header("5. RESOURCE METRICS (CPU/RAM/GPU) — NEW FORMAT")

    rdf = _resource_subset_new(df)
    if rdf.empty:
        print("⚠️ No NEW-format resource metrics found in loaded CSVs.")
        print("Tip: Ensure your results CSVs were generated with the updated run_experiment.py that logs CPU/RAM/GPU.")
        return

    def p95(x): return x.quantile(0.95)

    res = rdf.groupby("model").agg(
        samples=("question_id", "count") if "question_id" in rdf.columns else ("model", "size"),

        # latency
        avg_time_s=("generation_time", "mean"),
        p95_time_s=("generation_time", p95),

        # CPU
        avg_cpu_proc_percent=("cpu_proc_percent", "mean"),
        p95_cpu_proc_percent=("cpu_proc_percent", p95),
        avg_cpu_system_percent=("cpu_system_percent", "mean"),

        # RAM
        avg_ram_rss_mb=("ram_proc_rss_mb", "mean"),
        peak_ram_rss_mb=("ram_proc_rss_mb", "max"),
        avg_ram_system_percent=("ram_system_percent", "mean"),

        # GPU (may be NaN if no NVIDIA GPU)
        avg_gpu_util_percent=("gpu_util_percent", "mean"),
        p95_gpu_util_percent=("gpu_util_percent", p95),
        peak_gpu_mem_used_mb=("gpu_mem_used_mb", "max"),
        gpu_mem_total_mb=("gpu_mem_total_mb", "max"),
        peak_gpu_mem_percent=("gpu_mem_percent", "max"),
    )

    # rounding
    round_cols = {
        "avg_time_s": 4, "p95_time_s": 4,
        "avg_cpu_proc_percent": 2, "p95_cpu_proc_percent": 2, "avg_cpu_system_percent": 2,
        "avg_ram_rss_mb": 2, "peak_ram_rss_mb": 2, "avg_ram_system_percent": 2,
        "avg_gpu_util_percent": 2, "p95_gpu_util_percent": 2,
        "peak_gpu_mem_used_mb": 2, "gpu_mem_total_mb": 2, "peak_gpu_mem_percent": 2,
    }
    for c, nd in round_cols.items():
        if c in res.columns:
            res[c] = res[c].round(nd)

    cols = [
        "samples",
        "avg_time_s", "p95_time_s",
        "avg_cpu_proc_percent", "p95_cpu_proc_percent", "avg_cpu_system_percent",
        "avg_ram_rss_mb", "peak_ram_rss_mb", "avg_ram_system_percent",
        "avg_gpu_util_percent", "p95_gpu_util_percent",
        "peak_gpu_mem_used_mb", "gpu_mem_total_mb", "peak_gpu_mem_percent",
    ]
    cols = [c for c in cols if c in res.columns]
    print(res[cols].to_string())

def generate_resource_charts(df):
    rdf = _resource_subset_new(df)
    if rdf.empty:
        return

    sns.set_theme(style="whitegrid")
    model_palette = get_model_palette(rdf)

    # aggregate per model for plotting
    agg = rdf.groupby("model").agg(
        peak_ram_rss_mb=("ram_proc_rss_mb", "max"),
        avg_cpu_proc_percent=("cpu_proc_percent", "mean"),
        peak_gpu_mem_used_mb=("gpu_mem_used_mb", "max"),
        avg_gpu_util_percent=("gpu_util_percent", "mean"),
        p95_time_s=("generation_time", lambda x: x.quantile(0.95)),
    ).reset_index()

    # 1) Peak RAM chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=agg, x="model", y="peak_ram_rss_mb", hue="model", palette=model_palette, legend=False)
    plt.title("Peak RAM (Process RSS) by Model")
    plt.ylabel("Peak RAM (MB)")
    plt.xlabel("Model")
    out_path = os.path.join(results_dir, "peak_ram_by_model.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\n📊 Resource chart saved to: {out_path}")

    # 2) Avg CPU chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=agg, x="model", y="avg_cpu_proc_percent", hue="model", palette=model_palette, legend=False)
    plt.title("Average CPU (Process %) by Model")
    plt.ylabel("Avg CPU (%)")
    plt.xlabel("Model")
    out_path = os.path.join(results_dir, "avg_cpu_by_model.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Resource chart saved to: {out_path}")

    # 3) p95 latency chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=agg, x="model", y="p95_time_s", hue="model", palette=model_palette, legend=False)
    plt.title("p95 Generation Time by Model")
    plt.ylabel("p95 time (s)")
    plt.xlabel("Model")
    out_path = os.path.join(results_dir, "p95_time_by_model.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Resource chart saved to: {out_path}")

    # 4) GPU charts only if any GPU data exists
    if agg["peak_gpu_mem_used_mb"].notna().any():
        plt.figure(figsize=(8, 5))
        sns.barplot(data=agg, x="model", y="peak_gpu_mem_used_mb", hue="model", palette=model_palette, legend=False)
        plt.title("Peak GPU VRAM Used by Model")
        plt.ylabel("Peak VRAM Used (MB)")
        plt.xlabel("Model")
        out_path = os.path.join(results_dir, "peak_gpu_vram_used_by_model.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"📊 Resource chart saved to: {out_path}")

    if agg["avg_gpu_util_percent"].notna().any():
        plt.figure(figsize=(8, 5))
        sns.barplot(data=agg, x="model", y="avg_gpu_util_percent", hue="model", palette=model_palette, legend=False)
        plt.title("Average GPU Utilization by Model")
        plt.ylabel("Avg GPU Util (%)")
        plt.xlabel("Model")
        out_path = os.path.join(results_dir, "avg_gpu_util_by_model.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"📊 Resource chart saved to: {out_path}")

def generate_charts(df):
    """Accuracy by complexity chart."""
    try:
        sns.set_theme(style="whitegrid")
        model_palette = get_model_palette(df)

        if "complexity" not in df.columns:
            print("\n⚠️  No 'complexity' column found. Skipping accuracy-by-complexity chart.")
            return

        plt.figure(figsize=(10, 6))

        acc_by_comp = df.groupby(["model", "complexity"], observed=False)["is_correct"].mean().reset_index()
        acc_by_comp["is_correct"] *= 100

        sns.barplot(data=acc_by_comp, x="complexity", y="is_correct", hue="model", palette=model_palette)
        plt.title("Model Accuracy by Query Complexity")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Complexity Level")
        plt.ylim(0, 100)

        output_path = os.path.join(results_dir, "benchmark_chart.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"\n📊 Chart saved to: {output_path}")

    except ImportError:
        print("\n⚠️  Matplotlib/Seaborn not found. Skipping chart generation.")
    except Exception as e:
        print(f"\n⚠️  Could not generate chart: {e}")

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} total records from {results_dir}")
    print(f"CSV sources: {sorted(df['__source_file'].dropna().unique().tolist())}")

    analyze_difficulty_distribution(df)
    analyze_performance(df)
    analyze_dialect_robustness(df)
    head_to_head_comparison(df)

    analyze_resource_metrics(df)
    generate_resource_charts(df)

    generate_charts(df)
