import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
results_dir = os.path.join(project_root, "results")

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
    full_df['complexity'] = pd.Categorical(
        full_df['complexity'],
        categories=complexity_order,
        ordered=True
    )

    return full_df

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def analyze_difficulty_distribution(df):
    print_header("1. COMPLEXITY DISTRIBUTION (Count of Questions)")

    # Group by Dataset and Complexity
    dist = df.groupby(['dataset', 'complexity'], observed=False).size().unstack(fill_value=0)

    # Calculate Total
    dist['Total'] = dist.sum(axis=1)

    print(dist)
    print("\nObservation: This shows how balanced your test set was per dataset.")

def analyze_performance(df):
    print_header("2. MODEL PERFORMANCE (Accuracy & Latency)")

    # Group by Model and Dataset
    # We calculate:
    # - Accuracy: Mean of 'is_correct'
    # - SQLite Exec: Mean of 'pred_exec_success_sqlite'
    # - PG Exec: Mean of 'pred_exec_success_pg'
    # - Latency: Mean of 'generation_time'

    metrics = df.groupby(['model', 'dataset']).agg({
        'is_correct': 'mean',
        'pred_exec_success_sqlite': 'mean',
        'pred_exec_success_pg': 'mean',
        'generation_time': 'mean',
        'question_id': 'count' # Count total samples
    }).rename(columns={'question_id': 'samples'})

    # Convert fractions to percentages for readability
    metrics['Accuracy %'] = (metrics['is_correct'] * 100).round(2)
    metrics['SQLite Exec %'] = (metrics['pred_exec_success_sqlite'] * 100).round(2)
    metrics['Postgres Exec %'] = (metrics['pred_exec_success_pg'] * 100).round(2)
    metrics['Avg Time (s)'] = metrics['generation_time'].round(4)

    # Select clean columns to display
    display_cols = ['samples', 'Accuracy %', 'SQLite Exec %', 'Postgres Exec %', 'Avg Time (s)']
    print(metrics[display_cols])

def analyze_dialect_robustness(df):
    print_header("3. DIALECT ROBUSTNESS (Postgres vs SQLite)")

    # Logic: If it executes in SQLite but fails in Postgres,
    # the model is likely generating SQLite-specific syntax (hallucination).

    # Filter only rows where code executed in SQLite
    sqlite_valid = df[df['pred_exec_success_sqlite'] == True].copy()

    if sqlite_valid.empty:
        print("No queries executed successfully in SQLite to compare.")
        return

    # Group by model
    robustness = sqlite_valid.groupby('model').agg({
        'pred_exec_success_pg': 'mean'
    })

    robustness['PG Compatibility %'] = (robustness['pred_exec_success_pg'] * 100).round(2)

    print("Of queries that worked in SQLite, what % also worked in Postgres?")
    print(robustness[['PG Compatibility %']])
    print("\nNote: Lower percentages indicate the model is relying on SQLite-specific loose typing or syntax.")

def head_to_head_comparison(df):
    print_header("4. HEAD-TO-HEAD SUMMARY")

    # Aggregate globally by model
    summary = df.groupby('model').agg({
        'is_correct': 'mean',
        'generation_time': 'mean',
        'pred_exec_success_sqlite': 'mean'
    })

    print(f"{'Model':<20} | {'Accuracy':<10} | {'Exec Rate':<10} | {'Time/Query':<10}")
    print("-" * 60)

    for model, row in summary.iterrows():
        acc = f"{row['is_correct']*100:.2f}%"
        exe = f"{row['pred_exec_success_sqlite']*100:.2f}%"
        time = f"{row['generation_time']:.4f}s"
        print(f"{model:<20} | {acc:<10} | {exe:<10} | {time:<10}")

def generate_charts(df):
    """Generates a chart image if libraries are available"""
    try:
        sns.set_theme(style="whitegrid")

        # Plot 1: Accuracy by Complexity
        plt.figure(figsize=(10, 6))

        # Calculate accuracy by model and complexity
        acc_by_comp = df.groupby(['model', 'complexity'], observed=False)['is_correct'].mean().reset_index()
        acc_by_comp['is_correct'] *= 100

        sns.barplot(data=acc_by_comp, x='complexity', y='is_correct', hue='model', palette="viridis")
        plt.title("Model Accuracy by Query Complexity")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Complexity Level")
        plt.ylim(0, 100)

        output_path = os.path.join(results_dir, "benchmark_chart.png")
        plt.savefig(output_path)
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
    generate_charts(df)
