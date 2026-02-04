import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys

# -------------------------------------------------------------------
# ⚙️ CONFIGURATION & SETTINGS
# -------------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Paths
import os

try:
    # This works if running as a script (e.g., python analyze.py)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
except NameError:
    # This works if running in Jupyter / Colab
    # Assuming you are in the root directory (e.g., /content)
    CURRENT_DIR = os.getcwd()
    PROJECT_ROOT = CURRENT_DIR

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(PROJECT_ROOT, "report_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Plotting Style (Academic/Paper quality)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE_NAME = "viridis"

def get_model_palette(df):
    """Generates a consistent color map for models."""
    models = sorted(df["model"].dropna().unique().tolist())
    colors = sns.color_palette(PALETTE_NAME, n_colors=len(models))
    return {m: c for m, c in zip(models, colors)}

def save_plot(filename):
    """Helper to save plots consistently."""
    out_path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"   🖼️  Saved plot: {filename}")
    plt.close()

# -------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------
def load_data():
    print(f"📂 Loading results from: {RESULTS_DIR}")
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

    if not files:
        print("❌ No CSV files found in 'results' directory.")
        print("   Run 'python experiments/run_experiment.py' first.")
        sys.exit(1)

    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"⚠️  Skipping unreadable file {f}: {e}")

    if not df_list:
        sys.exit(1)

    df_all = pd.concat(df_list, ignore_index=True)
    
    # --- FIX: Clean Complexity Labels ---
    # 1. Drop rows with empty/null complexity
    if "complexity" in df_all.columns:
        df_all = df_all[df_all["complexity"].notna() & (df_all["complexity"] != "")]
    
    # 2. Enforce Order (Removed "Unknown" from here)
    complexity_order = ["Easy", "Medium", "Hard", "Extra Hard"]
    df_all["complexity"] = pd.Categorical(
        df_all["complexity"], categories=complexity_order, ordered=True
    )

    print(f"✅ Loaded {len(df_all)} total rows from {len(files)} files.")
    return df_all
    
# -------------------------------------------------------------------
# 2. DATA AGGREGATION & CSV EXPORT
# -------------------------------------------------------------------
def analyze_and_export(df):
    print("\n" + "="*40)
    print("📊 CALCULATING METRICS")
    print("="*40)

    # --- A. Detailed Complexity Stats ---
    complexity_stats = df.groupby(["model", "dataset", "complexity"], observed=False).agg(
        samples=("question_id", "count"),
        accuracy=("is_correct", "mean"),
        sqlite_exec=("pred_exec_success_sqlite", "mean"),
        pg_exec=("pred_exec_success_pg", "mean"),
        avg_latency=("generation_time", "mean")
    ).reset_index()

    # Convert to %
    for col in ["accuracy", "sqlite_exec", "pg_exec"]:
        complexity_stats[col] *= 100
    
    complexity_stats.to_csv(os.path.join(OUT_DIR, "detailed_complexity_stats.csv"), index=False)
    print("   💾 Saved: detailed_complexity_stats.csv")

    # --- B. Runtime & Latency Stats ---
    runtime_stats = df.groupby(["model", "dataset"], observed=False).agg(
        total_questions=("question_id", "count"),
        total_runtime_sec=("generation_time", "sum"),
        avg_latency_sec=("generation_time", "mean")
    ).reset_index()
    
    runtime_stats.to_csv(os.path.join(OUT_DIR, "runtime_stats.csv"), index=False)
    print("   💾 Saved: runtime_stats.csv")

    # --- C. Overall Summary ---
    summary = df.groupby("model", observed=False).agg(
        total_samples=("question_id", "count"),
        overall_accuracy=("is_correct", "mean"),
        overall_exec_rate=("pred_exec_success_sqlite", "mean"),
        total_time_minutes=("generation_time", lambda x: x.sum() / 60)
    ).reset_index()

    summary["overall_accuracy"] *= 100
    summary["overall_exec_rate"] *= 100
    
    summary.to_csv(os.path.join(OUT_DIR, "overall_model_summary.csv"), index=False)
    print("   💾 Saved: overall_model_summary.csv")

    # --- D. Dialect Robustness ---
    # Filter: Queries that executed in SQLite -> Did they work in Postgres?
    sqlite_ok = df[df["pred_exec_success_sqlite"] == True].copy()
    if not sqlite_ok.empty:
        robustness = sqlite_ok.groupby("model", observed=False).agg(
            pg_compatibility=("pred_exec_success_pg", "mean")
        ).reset_index()
        robustness["pg_compatibility"] *= 100
        robustness.to_csv(os.path.join(OUT_DIR, "dialect_robustness.csv"), index=False)
        print("   💾 Saved: dialect_robustness.csv")
    else:
        robustness = pd.DataFrame()

    # Print Console Summary
    print("\n--- Quick Summary (Console) ---")
    print(summary[["model", "overall_accuracy", "overall_exec_rate", "total_time_minutes"]].to_string())

    return complexity_stats, runtime_stats, summary, robustness

# -------------------------------------------------------------------
# 3. PLOTTING FUNCTIONS
# -------------------------------------------------------------------
def generate_plots(df_comp, df_run, df_sum, df_rob):
    print("\n" + "="*40)
    print("🎨 GENERATING PLOTS")
    print("="*40)
    
    # Ensure consistent colors
    palette = get_model_palette(df_sum)

    # --- Plot 1: Overall Performance (Acc vs Exec) ---
    summary_melt = df_sum.melt(
        id_vars=["model"],
        value_vars=["overall_accuracy", "overall_exec_rate"],
        var_name="Metric",
        value_name="Percentage"
    )
    summary_melt["Metric"] = summary_melt["Metric"].replace({
        "overall_accuracy": "Exact Match Accuracy",
        "overall_exec_rate": "Execution Success Rate"
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(data=summary_melt, x="Metric", y="Percentage", hue="model", palette=palette)
    plt.title("Overall Performance: Accuracy vs. Stability", fontsize=14, weight='bold')
    plt.ylabel("Percentage (%)")
    plt.xlabel("")
    plt.legend(title="Model")
    save_plot("plot_overall_performance.png")

    # --- Plot 2: Dialect Robustness ---
    if not df_rob.empty:
        plt.figure(figsize=(7, 5))
        ax = sns.barplot(data=df_rob, x="model", y="pg_compatibility", palette=palette, hue="model", legend=False)
        plt.title("PostgreSQL Compatibility Score", fontsize=14, weight='bold')
        plt.ylabel("Compatibility (%)")
        plt.xlabel("Model")
        for i in ax.containers:
            ax.bar_label(i, fmt='%.1f%%', padding=3)
        plt.ylim(0, df_rob["pg_compatibility"].max() + 10)
        save_plot("plot_dialect_robustness.png")

    # --- Plot 3: Latency by Dataset ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_run, x="dataset", y="avg_latency_sec", hue="model", palette=palette)
    plt.title("Average Inference Latency per Query", fontsize=14, weight='bold')
    plt.ylabel("Latency (seconds)")
    plt.xlabel("Dataset")
    plt.legend(title="Model")
    save_plot("plot_latency_by_dataset.png")

    # --- Plot 4: Complexity Breakdown (Per Dataset) ---
    datasets = df_comp["dataset"].unique()
    for ds in datasets:
        ds_data = df_comp[df_comp["dataset"] == ds]
        
        # Skip empty datasets
        if ds_data.empty or ds_data["samples"].sum() == 0:
            continue

        plt.figure(figsize=(8, 5))
        sns.barplot(data=ds_data, x="complexity", y="accuracy", hue="model", palette=palette)
        
        title_map = {
            "spider": "Spider Dataset (General Reasoning)",
            "atis": "ATIS Dataset (Flight Booking)",
            "geography": "Geography Dataset",
            "custom": "Custom Dataset"
        }
        plt.title(title_map.get(ds, f"{ds.capitalize()} Dataset"), fontsize=14, weight='bold')
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Complexity")
        plt.legend(title="Model")
        plt.ylim(0, 100)  # Standardize y-axis
        
        save_plot(f"plot_{ds}_complexity.png")

# -------------------------------------------------------------------
# 4. RESOURCE ANALYSIS (Optional)
# -------------------------------------------------------------------
def analyze_resources(df):
    # Check if resource cols exist (from updated run_experiment.py)
    cols_needed = ["total_runtime_s", "peak_ram_rss_mb", "peak_gpu_vram_allocated_mb"]
    if not all(col in df.columns for col in cols_needed):
        print("\n⚠️  Resource columns missing. Skipping resource plots.")
        return

    # Filter for rows that actually have runtime data (last row of a run usually)
    res_df = df.dropna(subset=cols_needed).groupby("model", observed=False).agg({
        "total_runtime_s": "max",
        "peak_ram_rss_mb": "max",
        "peak_gpu_vram_allocated_mb": "max"
    }).reset_index()

    if res_df.empty:
        return

    print("\n" + "="*40)
    print("🔋 RESOURCE ANALYSIS")
    print("="*40)
    palette = get_model_palette(res_df)

    # Plot Total Runtime
    plt.figure(figsize=(7, 5))
    sns.barplot(data=res_df, x="model", y="total_runtime_s", hue="model", palette=palette, legend=False)
    plt.title("Total Benchmark Runtime", fontsize=14, weight='bold')
    plt.ylabel("Time (seconds)")
    save_plot("total_runtime_comparison.png")

    print(res_df.to_string())

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load
    full_df = load_data()

    # 2. Analyze & Export CSVs
    df_comp, df_run, df_sum, df_rob = analyze_and_export(full_df)

    # 3. Generate Charts
    generate_plots(df_comp, df_run, df_sum, df_rob)

    # 4. Resource Analysis
    analyze_resources(full_df)

    print(f"\n✅ DONE! Check the '{OUT_DIR}' folder for all reports and images.")
