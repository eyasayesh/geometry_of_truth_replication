#!/usr/bin/env python3
"""
Visualize cross-model probe transfer results as grouped bar charts.

For each (probe_model, activation_model, layer, train_group) combination found
in the cross-model probes directory, produces a bar chart with activation
datasets on the x-axis and one bar per probe type (LR, MM, MM+cov).

Results are read from:
  {cross_model_probes_dir}/{probe_model}__{activation_model}/layer_{layer}/results.json

Figures are written to:
  {figures_dir}/{probe_model}__{activation_model}/layer_{layer}/{train_key}.png

Uses the same config format as cross_model_probe_pipeline.py.

Example:
    python scripts/visualize_cross_model_probes.py \\
        --config data/configs/probes/cross_model_probe_config.json \\
        --cross_model_probes_dir /scratch/geometry_of_truth/cross_model_probes \\
        --figures_dir figures/cross_model_probes
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.visualize_probes import plot_probe_bar
from src.models import MODEL_REGISTRY
from src.probes import load_probe_results


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pair_title(probe_model: str, activation_model: str) -> str:
    probe_display = MODEL_REGISTRY[probe_model].display_name if probe_model in MODEL_REGISTRY else probe_model
    act_display   = MODEL_REGISTRY[activation_model].display_name if activation_model in MODEL_REGISTRY else activation_model
    return f"probe: {probe_display}  →  act: {act_display}"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize cross-model probe transfer results")
    parser.add_argument(
        "--config", required=True,
        help="Cross-model probe config JSON (same format as cross_model_probe_pipeline.py)",
    )
    parser.add_argument(
        "--cross_model_probes_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/cross_model_probes",
        help="Root directory where cross-model probe results are saved",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures/cross_model_probes",
        help="Root directory for output PNG figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    probe_datasets      = cfg["probe_datasets"]
    activation_datasets = cfg["activation_datasets"]

    for pair in cfg["pairs"]:
        probe_model      = pair["probe_model"]
        activation_model = pair["activation_model"]
        layers           = pair["layers"]
        pair_key         = f"{probe_model}__{activation_model}"
        title_prefix     = pair_title(probe_model, activation_model)

        for layer in layers:
            results_path = os.path.join(
                args.cross_model_probes_dir, pair_key, f"layer_{layer}", "results.json"
            )
            if not os.path.exists(results_path):
                print(f"WARNING: missing results for {pair_key} layer {layer} — skipping ({results_path})")
                continue

            df      = load_probe_results(results_path)
            fig_dir = os.path.join(args.figures_dir, pair_key, f"layer_{layer}")
            os.makedirs(fig_dir, exist_ok=True)

            for train_key in probe_datasets:
                if train_key not in df.index.get_level_values("train_group"):
                    print(f"WARNING: no results for train_group '{train_key}' in {pair_key} layer {layer} — skipping")
                    continue

                fig = plot_probe_bar(df, train_key, activation_datasets, pair_key, layer)
                # Replace the generic title with one that names both models clearly
                fig.update_layout(title=dict(
                    text=f"{title_prefix} — layer {layer} — train: {train_key}"
                ))

                out_path = os.path.join(fig_dir, f"{train_key}.png")
                fig.write_image(out_path)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
