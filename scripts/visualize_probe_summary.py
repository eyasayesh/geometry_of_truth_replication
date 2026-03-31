#!/usr/bin/env python3
"""
Probe summary visualization: mean accuracy over test datasets with std error
bars, individual test-dataset score points, and connecting lines.

Two modes controlled by --mode:

  train_groups — x-axis = train groups, one figure per (model, layer).
                 Shows how probe performance varies across training sets.

  layers       — x-axis = layers, one figure per (model, train_group).
                 Shows how probe performance evolves across depth.

Probe columns are auto-detected from results.json, so the same script works
for both regular probes (lr, mm, mm_cov) and original probes (lr, mm,
mm_iid, ccs).

Figures are written to:
  train_groups mode: {figures_dir}/{model}/layer_{layer}/train_groups_summary.png
  layers mode:       {figures_dir}/{model}/{train_group}/layers_summary.png

Uses the same config format as probe_pipeline.py / original_probe_pipeline.py.

Example:
    python scripts/visualize_probe_summary.py \\
        --config data/configs/probes/probe_viz.json \\
        --mode train_groups \\
        --probes_dir /scratch/geometry_of_truth/probes

    python scripts/visualize_probe_summary.py \\
        --config data/configs/probes/probe_viz.json \\
        --mode layers \\
        --probes_dir /scratch/geometry_of_truth/probes_original \\
        --figures_dir figures/probe_summary_original
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MODEL_REGISTRY
from src.plot_utils import detect_probe_cols, load_probe_scores, plot_probe_summary


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_train_groups_mode(cfg: dict, probes_dir: str, figures_dir: str) -> None:
    """One figure per (model, layer); x-axis = train groups."""
    train_groups = ["+".join(g) for g in cfg.get("train_groups", [])]

    for model_name, model_cfg in cfg["models"].items():
        display = MODEL_REGISTRY[model_name].display_name if model_name in MODEL_REGISTRY else model_name

        for layer in model_cfg["layers"]:
            results_path = os.path.join(probes_dir, model_name, f"layer_{layer}", "results.json")
            if not os.path.exists(results_path):
                print(f"WARNING: missing {results_path} — skipping")
                continue

            full_df    = load_probe_scores(results_path)
            probe_cols = detect_probe_cols(full_df)
            present    = set(full_df.index.get_level_values("train_group"))

            data = {
                tg: full_df.loc[tg]
                for tg in train_groups
                if tg in present
            }
            if not data:
                print(f"WARNING: no matching train groups in {results_path} — skipping")
                continue

            title = f"{display} — layer {layer}"
            fig   = plot_probe_summary(data, probe_cols, title=title)

            fig_dir  = os.path.join(figures_dir, model_name, f"layer_{layer}")
            out_path = os.path.join(fig_dir, "train_groups_summary.png")
            os.makedirs(fig_dir, exist_ok=True)
            fig.write_image(out_path)
            print(f"Saved: {out_path}")


def run_layers_mode(cfg: dict, probes_dir: str, figures_dir: str) -> None:
    """One figure per (model, train_group); x-axis = layers."""
    train_groups = ["+".join(g) for g in cfg.get("train_groups", [])]

    for model_name, model_cfg in cfg["models"].items():
        display = MODEL_REGISTRY[model_name].display_name if model_name in MODEL_REGISTRY else model_name
        layers  = model_cfg["layers"]

        # Load all available layers
        layer_dfs: dict[int, object] = {}
        for layer in layers:
            results_path = os.path.join(probes_dir, model_name, f"layer_{layer}", "results.json")
            if not os.path.exists(results_path):
                print(f"WARNING: missing {results_path} — skipping layer {layer}")
                continue
            layer_dfs[layer] = load_probe_scores(results_path)

        if not layer_dfs:
            continue

        for tg in train_groups:
            data: dict[str, object] = {}
            for layer, df in layer_dfs.items():
                present = set(df.index.get_level_values("train_group"))
                if tg in present:
                    data[f"layer {layer}"] = df.loc[tg]

            if not data:
                continue

            probe_cols = detect_probe_cols(next(iter(data.values())))
            title      = f"{display} — train: {tg}"
            fig        = plot_probe_summary(data, probe_cols, title=title)

            fig_dir  = os.path.join(figures_dir, model_name, tg)
            out_path = os.path.join(fig_dir, "layers_summary.png")
            os.makedirs(fig_dir, exist_ok=True)
            fig.write_image(out_path)
            print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Probe summary visualization")
    parser.add_argument("--config", required=True, help="Probe pipeline config JSON")
    parser.add_argument(
        "--mode",
        choices=["train_groups", "layers"],
        default="train_groups",
        help=(
            "x-axis grouping: "
            "'train_groups' = one fig per (model, layer); "
            "'layers' = one fig per (model, train_group)"
        ),
    )
    parser.add_argument(
        "--probes_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes",
        help="Root directory containing probe results",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures/probe_summary",
        help="Root directory for output PNG figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.mode == "train_groups":
        run_train_groups_mode(cfg, args.probes_dir, args.figures_dir)
    else:
        run_layers_mode(cfg, args.probes_dir, args.figures_dir)


if __name__ == "__main__":
    main()
