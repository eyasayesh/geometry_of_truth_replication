#!/usr/bin/env python3
"""
Visualize probe accuracy results as grouped bar charts.

For each (model, layer, train_group) combination found in the probes directory,
produces a bar chart with test datasets on the x-axis and one bar per probe type
(LR, MM, MM+cov). Test datasets are separated by visual gaps.

Results are read from:
  {probes_dir}/{model}/layer_{layer}/results.json

Figures are written to:
  {figures_dir}/{model}/layer_{layer}/{train_key}.png

Uses the same config format as probe_pipeline.py.

Example:
    python scripts/visualize_probes.py \\
        --config data/configs/probes/probe_config.json \\
        --probes_dir /scratch/geometry_of_truth/probes \\
        --figures_dir figures/probes
"""

import argparse
import json
import os
import sys

import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MODEL_REGISTRY
from src.probes import load_probe_results


def plot_probe_bar(df, train_key, test_datasets, model_name, layer):
    """
    Grouped bar chart for one (model, layer, train_group) combo.

    X-axis: test datasets separated by blank spacer categories.
    Bars: LR, MM, MM+cov grouped per dataset.
    """
    # Build x-axis with unique zero-width-space spacers between real categories
    x_cats = []
    y = {"lr": [], "mm": [], "mm_cov": []}

    for i, ds in enumerate(test_datasets):
        if i > 0:
            spacer = "\u200b" * i  # zero-width spaces — visually blank, unique per gap
            x_cats.append(spacer)
            for k in y:
                y[k].append(None)

        x_cats.append(ds)
        try:
            row = df.loc[(train_key, ds)]
            y["lr"].append(float(row["lr"]))
            y["mm"].append(float(row["mm"]))
            y["mm_cov"].append(float(row["mm_cov"]))
        except KeyError:
            for k in y:
                y[k].append(None)

    model_display = MODEL_REGISTRY[model_name].display_name if model_name in MODEL_REGISTRY else model_name
    title = f"{model_display} — layer {layer} — train: {train_key}"

    fig = go.Figure([
        go.Bar(name="LR",     x=x_cats, y=y["lr"],     marker_color="#1f77b4"),
        go.Bar(name="MM",     x=x_cats, y=y["mm"],     marker_color="#ff7f0e"),
        go.Bar(name="MM+cov", x=x_cats, y=y["mm_cov"], marker_color="#2ca02c"),
    ])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        barmode="group",
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=11),
        ),
        yaxis=dict(title="Accuracy", range=[0, 1], tickformat=".0%"),
        shapes=[dict(
            type="line", x0=0, x1=1, y0=0.5, y1=0.5,
            xref="paper", yref="y",
            line=dict(color="red", width=1.5, dash="dash"),
        )],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=450,
        margin=dict(b=120),
    )
    return fig


def load_config(path):
    with open(path) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize probe accuracy results")
    parser.add_argument(
        "--config", required=True,
        help="Probe pipeline config JSON (same format as probe_pipeline.py)",
    )
    parser.add_argument(
        "--probes_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes",
        help="Root directory where probe results are saved",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures/probes",
        help="Root directory for output PNG figures (default: figures/probes)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    train_groups = [tuple(g) for g in cfg.get("train_groups", [])]
    test_datasets = cfg.get("test_datasets", [])

    for model_name, model_cfg in cfg["models"].items():
        for layer in model_cfg["layers"]:
            results_path = os.path.join(args.probes_dir, model_name, f"layer_{layer}", "results.json")

            if not os.path.exists(results_path):
                print(f"WARNING: missing results for {model_name} layer {layer} — skipping ({results_path})")
                continue

            df = load_probe_results(results_path)
            fig_dir = os.path.join(args.figures_dir, model_name, f"layer_{layer}")
            os.makedirs(fig_dir, exist_ok=True)

            for group in train_groups:
                train_key = "+".join(group)
                if train_key not in df.index.get_level_values("train_group"):
                    print(f"WARNING: no results for train_group '{train_key}' in {model_name} layer {layer} — skipping")
                    continue

                fig = plot_probe_bar(df, train_key, test_datasets, model_name, layer)
                out_path = os.path.join(fig_dir, f"{train_key}.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
