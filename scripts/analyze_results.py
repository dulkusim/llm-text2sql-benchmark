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
    # stable ordering so colors don't swap between runs
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
            df_list.append(temp_df)
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")

    if not df_list:
        sys.exit(1)

    full_df = pd.concat(df_list, ignore_index=True)

    # Enforce Complexity Order
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
        "generation_time": "mean",
        "question_id": "count"
    }).rename(columns={"question_id": "samples"})

    metrics["Accuracy %"] = (metrics["is_correct"] * 100).round(2)
    metrics["SQLite Exec %"] = (metrics["pred_exec_success_sqlite"] * 100).round(2)
    metrics["Postgres Exec %"] = (metrics["pred_exec_success_pg"] * 100).round(2)
    metrics["Avg Time (s)"] = metrics["generation_time"].round(4)

    display_cols = ["samples", "Accuracy %", "SQLite Exec %", "Postgres Exec %", "Avg Time (s)"]
    print(metrics[display_cols].to_string())

def analyze_dialect_robustness(df):
    print_header("3. DIALECT ROBUSTNESS (Postgres vs SQLite)")

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
# Resource metrics analysis (works even with mixed CSVs)
# ---------------------------
RESOURCE_COLS = [
    "total_runtime_s",
    "peak_ram_rss_mb",
    "peak_cpu_percent_process",
    "peak_gpu_vram_allocated_mb",
    "peak_gpu_vram_reserved_mb",
    "cpu_ram_available",
    "gpu_available",
]

def _resource_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only rows that contain resource metrics.
    This fixes the case where results_dir has mixed CSVs (old ones without resource cols).
    """
    # If the columns don't exist at all, return empty
    for c in ["model"] + RESOURCE_COLS:
        if c not in df.columns:
            return df.iloc[0:0].copy()

    sub = df.copy()

    # Ensure numeric where possible
    for c in RESOURCE_COLS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # Keep only rows where runtime exists (means the row came from a run with Option B)
    sub = sub[sub["total_runtime_s"].notna()]
    return sub

def analyze_resource_metrics(df):
    print_header("5. RESOURCE METRICS (CPU/RAM/GPU)")

    rdf = _resource_subset(df)
    if rdf.empty:
        print("⚠️ No resource metrics found in the loaded CSVs.")
        print("Tip: Keep the new Option-B CSVs in results/, or delete the old CSVs to avoid mixing.")
        return

    res = rdf.groupby("model").agg({
        "total_runtime_s": "max",
        "peak_ram_rss_mb": "max",
        "peak_cpu_percent_process": "max",
        "peak_gpu_vram_allocated_mb": "max",
        "peak_gpu_vram_reserved_mb": "max",
        "cpu_ram_available": "max",
        "gpu_available": "max",
        "question_id": "count" if "question_id" in rdf.columns else "size"
    }).rename(columns={"question_id": "samples"})

    res["total_runtime_s"] = res["total_runtime_s"].round(2)
    res["peak_ram_rss_mb"] = res["peak_ram_rss_mb"].round(2)
    res["peak_cpu_percent_process"] = res["peak_cpu_percent_process"].round(1)
    res["peak_gpu_vram_allocated_mb"] = res["peak_gpu_vram_allocated_mb"].round(2)
    res["peak_gpu_vram_reserved_mb"] = res["peak_gpu_vram_reserved_mb"].round(2)

    cols = [
        "samples",
        "total_runtime_s",
        "peak_ram_rss_mb",
        "peak_cpu_percent_process",
        "peak_gpu_vram_allocated_mb",
        "peak_gpu_vram_reserved_mb",
        "cpu_ram_available",
        "gpu_available",
    ]
    print(res[cols].to_string())

def generate_resource_charts(df):
    needed = {"model", "peak_ram_rss_mb", "peak_gpu_vram_allocated_mb", "total_runtime_s"}
    if not needed.issubset(set(df.columns)):
        return

    rdf = _resource_subset(df)
    if rdf.empty:
        return

    model_palette = get_model_palette(rdf)

    res = rdf.groupby("model").agg({
        "peak_ram_rss_mb": "max",
        "peak_gpu_vram_allocated_mb": "max",
        "total_runtime_s": "max",
    }).reset_index()

    # RAM chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=res, x="model", y="peak_ram_rss_mb", hue="model", palette=model_palette, legend=False)
    plt.title("Peak RAM (RSS) by Model")
    plt.ylabel("Peak RAM (MB)")
    plt.xlabel("Model")
    out_path = os.path.join(results_dir, "peak_ram_by_model.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\n📊 Resource chart saved to: {out_path}")

    # GPU chart (only if there are non-null values)
    if res["peak_gpu_vram_allocated_mb"].notna().any():
        plt.figure(figsize=(8, 5))
        sns.barplot(data=res, x="model", y="peak_gpu_vram_allocated_mb", hue="model", palette=model_palette, legend=False)
        plt.title("Peak GPU VRAM Allocated by Model")
        plt.ylabel("Peak GPU VRAM Allocated (MB)")
        plt.xlabel("Model")
        out_path = os.path.join(results_dir, "peak_gpu_vram_by_model.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"📊 Resource chart saved to: {out_path}")

    # Runtime chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=res, x="model", y="total_runtime_s", hue="model", palette=model_palette, legend=False)
    plt.title("Total Runtime by Model")
    plt.ylabel("Runtime (s)")
    plt.xlabel("Model")
    out_path = os.path.join(results_dir, "runtime_by_model.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Resource chart saved to: {out_path}")

def generate_charts(df):
    """Generates a chart image if libraries are available"""
    try:
        sns.set_theme(style="whitegrid")
        model_palette = get_model_palette(df)

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

    analyze_difficulty_distribution(df)
    analyze_performance(df)
    analyze_dialect_robustness(df)
    head_to_head_comparison(df)

    analyze_resource_metrics(df)
    generate_resource_charts(df)

    generate_charts(df)
