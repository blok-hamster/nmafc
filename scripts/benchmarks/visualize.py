"""Visualization and dashboard generator for benchmark results."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.benchmarks.evaluator import ScoredResult


def generate_visualizations(results: list[ScoredResult], output_dir: str = "charts") -> str:
    """Generate Plotly interactive HTML dashboard and Markdown summary report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = [
        {
            "Framework": r.framework,
            "Case_ID": r.case_id,
            "Category": r.category,
            "Accuracy_Score": r.accuracy_score,
            "Context_Collision": 1 if r.context_collision else 0,
            "Latency_Sec": r.latency_sec,
            "Token_Count": r.token_count,
            "Cost_USD": r.cost_usd,
        }
        for r in results
    ]

    df = pd.DataFrame(data)

    # Aggregated Summary by Framework
    summary = df.groupby("Framework").agg(
        Mean_Accuracy=("Accuracy_Score", "mean"),
        Collision_Rate=("Context_Collision", "mean"),
        Mean_Latency=("Latency_Sec", "mean"),
        Avg_Tokens=("Token_Count", "mean"),
        Total_Cost=("Cost_USD", "sum")
    ).reset_index()

    # 1. Bar Chart: Accuracy by Framework
    fig_acc = px.bar(
        summary,
        x="Framework",
        y="Mean_Accuracy",
        color="Framework",
        title="100-Case LoCoMo Benchmark: Memory Accuracy Score (Higher is Better)",
        text_auto=".2f",
        template="plotly_dark",
    )

    # 2. Bar Chart: Context Collision Rate
    fig_coll = px.bar(
        summary,
        x="Framework",
        y="Collision_Rate",
        color="Framework",
        title="Context Collision Rate (% of Contradictory Hallucinations - Lower is Better)",
        text_auto=".2%",
        template="plotly_dark",
    )

    # 3. Scatter Plot: Latency vs Cost Efficiency Frontier
    fig_eff = px.scatter(
        summary,
        x="Mean_Latency",
        y="Total_Cost",
        color="Framework",
        size="Mean_Accuracy",
        text="Framework",
        title="Latency vs. Cost Efficiency Frontier (Size = Accuracy)",
        template="plotly_dark",
        labels={"Mean_Latency": "Average Turn Latency (Seconds)", "Total_Cost": "Total Cost for 100 Runs ($ USD)"},
    )

    # Combine into a single dashboard HTML file
    dashboard_html = Path(output_dir) / "benchmark_dashboard.html"
    with open(dashboard_html, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Neuromorphic Memory Benchmark Dashboard</title></head><body style='background-color:#111; color:#fff; font-family:sans-serif; padding:20px;'>")
        f.write("<h1>🧠 100-Case LoCoMo & LongMemEval Benchmark Dashboard</h1>")
        f.write("<p>Comparing 5 Memory Frameworks using Azure OpenAI DeepSeek-V4-Pro</p>")
        f.write(fig_acc.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<br><hr><br>")
        f.write(fig_coll.to_html(full_html=False, include_plotlyjs=False))
        f.write("<br><hr><br>")
        f.write(fig_eff.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")

    # Generate Markdown Summary Report
    md_lines = [
        "# 📊 100-Case LoCoMo & LongMemEval Benchmark Summary Report",
        "",
        "| Framework | Mean Accuracy Score | Context Collision Rate | Avg Latency (s) | Avg Tokens/Turn | Total Cost ($ USD) |",
        "|---|---|---|---|---|---|",
    ]

    for _, row in summary.iterrows():
        md_lines.append(
            f"| **{row['Framework']}** | {row['Mean_Accuracy'] * 100:.1f}% | {row['Collision_Rate'] * 100:.1f}% | {row['Mean_Latency']:.2f}s | {int(row['Avg_Tokens'])} | ${row['Total_Cost']:.5f} |"
        )

    md_report = "\n".join(md_lines)
    with open(Path(output_dir) / "summary_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    return str(dashboard_html)
