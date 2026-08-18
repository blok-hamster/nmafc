"""Publication-ready benchmark visualization.

Generates Plotly charts matching the figures described in the NMAFC
exec summary and implementation methodology docs:
1. Accuracy by Question Type (grouped bar)
2. Token Cost comparison (bar)
3. Context Token Injection Size (bar — proves compression claim)
4. Latency comparison (box plot)
5. Combined HTML dashboard

Usage:
    python -m scripts.benchmarks.visualize --input results/locomo/results.json
    python -m scripts.benchmarks.visualize --input results/longmemeval/results.json --format svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_results(path: str) -> dict:
    """Load benchmark results JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy_by_type(results: dict, output_dir: Path, fmt: str = "html") -> None:
    """Grouped bar chart: accuracy by question type across arms."""
    rows = []
    for arm_name, arm_data in results["results"].items():
        accuracy = arm_data.get("accuracy", {})
        by_cat = accuracy.get("by_category", {})
        for cat, acc in by_cat.items():
            rows.append({"Arm": arm_name, "Category": cat, "Accuracy": acc})

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Category",
        y="Accuracy",
        color="Arm",
        barmode="group",
        title="Accuracy by Question Type",
        labels={"Accuracy": "Accuracy (LLM Judge)"},
    )
    fig.update_layout(
        yaxis_range=[0, 1],
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _save_fig(fig, output_dir / f"accuracy_by_type.{fmt}", fmt)


def plot_token_cost(results: dict, output_dir: Path, fmt: str = "html") -> None:
    """Bar chart: total token usage per arm."""
    rows = []
    for arm_name, arm_data in results["results"].items():
        ops = arm_data.get("operational", {})
        tokens = ops.get("tokens", {})
        rows.append({
            "Arm": arm_name,
            "Prompt Tokens": tokens.get("total_prompt", 0),
            "Completion Tokens": tokens.get("total_completion", 0),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Prompt", x=df["Arm"], y=df["Prompt Tokens"]))
    fig.add_trace(go.Bar(name="Completion", x=df["Arm"], y=df["Completion Tokens"]))
    fig.update_layout(
        barmode="stack",
        title="Total Token Usage by Arm",
        yaxis_title="Tokens",
        font=dict(size=14),
    )
    _save_fig(fig, output_dir / f"token_cost.{fmt}", fmt)


def plot_context_injection(results: dict, output_dir: Path, fmt: str = "html") -> None:
    """Bar chart: average context tokens injected per question."""
    rows = []
    for arm_name, arm_data in results["results"].items():
        ops = arm_data.get("operational", {})
        tokens = ops.get("tokens", {})
        rows.append({
            "Arm": arm_name,
            "Avg Context Tokens": tokens.get("avg_context_per_question", 0),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Arm",
        y="Avg Context Tokens",
        title="Average Context Injection Size per Question",
        labels={"Avg Context Tokens": "Context Tokens"},
        color="Arm",
    )
    fig.update_layout(font=dict(size=14), showlegend=False)
    _save_fig(fig, output_dir / f"context_injection.{fmt}", fmt)


def plot_latency(results: dict, output_dir: Path, fmt: str = "html") -> None:
    """Bar chart: latency comparison (avg, p50, p95)."""
    rows = []
    for arm_name, arm_data in results["results"].items():
        ops = arm_data.get("operational", {})
        latency = ops.get("latency_ms", {})
        rows.append({
            "Arm": arm_name,
            "Avg": latency.get("avg", 0),
            "P50": latency.get("p50", 0),
            "P95": latency.get("p95", 0),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Avg", x=df["Arm"], y=df["Avg"]))
    fig.add_trace(go.Bar(name="P50", x=df["Arm"], y=df["P50"]))
    fig.add_trace(go.Bar(name="P95", x=df["Arm"], y=df["P95"]))
    fig.update_layout(
        barmode="group",
        title="Response Latency by Arm",
        yaxis_title="Latency (ms)",
        font=dict(size=14),
    )
    _save_fig(fig, output_dir / f"latency.{fmt}", fmt)


def plot_dashboard(results: dict, output_dir: Path) -> None:
    """Combined HTML dashboard with all charts."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Accuracy by Category",
            "Context Injection Size",
            "Token Usage",
            "Latency",
        ),
    )

    arm_names = list(results["results"].keys())
    colors = px.colors.qualitative.Set2[:len(arm_names)]

    # Accuracy
    for i, (arm_name, arm_data) in enumerate(results["results"].items()):
        by_cat = arm_data.get("accuracy", {}).get("by_category", {})
        cats = sorted(by_cat.keys())
        fig.add_trace(
            go.Bar(name=arm_name, x=cats, y=[by_cat[c] for c in cats],
                   marker_color=colors[i], legendgroup=arm_name, showlegend=True),
            row=1, col=1,
        )

    # Context tokens
    for i, (arm_name, arm_data) in enumerate(results["results"].items()):
        ctx = arm_data.get("operational", {}).get("tokens", {}).get("avg_context_per_question", 0)
        fig.add_trace(
            go.Bar(name=arm_name, x=[arm_name], y=[ctx],
                   marker_color=colors[i], legendgroup=arm_name, showlegend=False),
            row=1, col=2,
        )

    # Token usage
    for i, (arm_name, arm_data) in enumerate(results["results"].items()):
        total = arm_data.get("operational", {}).get("tokens", {}).get("total", 0)
        fig.add_trace(
            go.Bar(name=arm_name, x=[arm_name], y=[total],
                   marker_color=colors[i], legendgroup=arm_name, showlegend=False),
            row=2, col=1,
        )

    # Latency
    for i, (arm_name, arm_data) in enumerate(results["results"].items()):
        lat = arm_data.get("operational", {}).get("latency_ms", {})
        fig.add_trace(
            go.Bar(name=arm_name, x=["Avg", "P50", "P95"],
                   y=[lat.get("avg", 0), lat.get("p50", 0), lat.get("p95", 0)],
                   marker_color=colors[i], legendgroup=arm_name, showlegend=False),
            row=2, col=2,
        )

    dataset = results.get("metadata", {}).get("dataset", "benchmark")
    fig.update_layout(
        height=800,
        title_text=f"NMAFC Benchmark Dashboard — {dataset.upper()}",
        font=dict(size=12),
        barmode="group",
    )

    _save_fig(fig, output_dir / "dashboard.html", "html")


def generate_summary_table(results: dict, output_dir: Path) -> None:
    """Generate markdown summary table."""
    metadata = results.get("metadata", {})
    lines = [
        f"# Benchmark Results: {metadata.get('dataset', 'unknown').upper()}",
        "",
        f"- **Provider:** {metadata.get('provider', 'unknown')}",
        f"- **Embedding:** {metadata.get('embedding', 'unknown')}",
        f"- **Judge:** {metadata.get('judge', 'unknown')}",
        f"- **Date:** {metadata.get('timestamp', 'unknown')}",
        f"- **Conversations:** {metadata.get('conversations_evaluated', 'unknown')}",
        f"- **Questions:** {metadata.get('questions_evaluated', 'unknown')}",
        # Retrieval settings belong on the figure, not only in the runner's
        # argv: two results files with identical arms can differ by these
        # alone, and a 2-hop run costs 3.5x the context of a 0-hop one.
        f"- **Spreading Activation (max_hops):** "
        f"{metadata.get('max_hops', 'default')}"
        f"{' — graph traversal disabled' if metadata.get('max_hops') == 0 else ''}",
        f"- **Retrieval threshold (theta):** {metadata.get('theta', 'unknown')}",
        "",
        "## Results",
        "",
        "| Arm | Accuracy | F1 | Avg Context | Avg Latency | Total Tokens |",
        "|-----|----------|-----|-------------|-------------|--------------|",
    ]

    for arm_name, arm_data in results["results"].items():
        acc = arm_data.get("accuracy", {}).get("overall", 0)
        f1 = arm_data.get("accuracy", {}).get("overall_f1", 0)
        ops = arm_data.get("operational", {})
        ctx = ops.get("tokens", {}).get("avg_context_per_question", 0)
        lat = ops.get("latency_ms", {}).get("avg", 0)
        total = ops.get("tokens", {}).get("total", 0)
        lines.append(
            f"| {arm_name} | {acc:.3f} | {f1:.3f} | {ctx:.0f} | {lat:.0f}ms | {total:,} |"
        )

    lines.append("")
    lines.append("## Per-Category Breakdown")
    lines.append("")

    # Build per-category table
    all_cats: set[str] = set()
    for arm_data in results["results"].values():
        all_cats.update(arm_data.get("accuracy", {}).get("by_category", {}).keys())

    if all_cats:
        header = "| Category |"
        separator = "|----------|"
        for arm_name in results["results"]:
            header += f" {arm_name} |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)

        for cat in sorted(all_cats):
            row = f"| {cat} |"
            for arm_data in results["results"].values():
                acc = arm_data.get("accuracy", {}).get("by_category", {}).get(cat, 0)
                row += f" {acc:.3f} |"
            lines.append(row)

    summary_path = output_dir / "SUMMARY.md"
    # Explicit encoding: Windows defaults to cp1252, which cannot encode the
    # em dashes in this file and silently writes replacement characters.
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary written to: {summary_path}")


def _save_fig(fig: go.Figure, path: Path, fmt: str) -> None:
    """Save figure in requested format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        fig.write_html(str(path), include_plotlyjs="cdn")
    elif fmt in ("svg", "png", "pdf"):
        fig.write_image(str(path), format=fmt, width=1200, height=600)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--input", required=True, help="Path to results.json")
    parser.add_argument("--format", default="html", choices=["html", "svg", "png", "pdf"])
    parser.add_argument("--output", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    results = load_results(args.input)
    output_dir = Path(args.output) if args.output else Path(args.input).parent

    print("Generating visualizations...")
    plot_accuracy_by_type(results, output_dir, args.format)
    plot_token_cost(results, output_dir, args.format)
    plot_context_injection(results, output_dir, args.format)
    plot_latency(results, output_dir, args.format)
    plot_dashboard(results, output_dir)
    generate_summary_table(results, output_dir)
    print(f"\nDone! All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
