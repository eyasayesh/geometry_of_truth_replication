#!/usr/bin/env python3
"""
Cross-model probe transfer summary visualization.

Compares probes from multiple source models evaluated on one activation model's
activations, against the in-domain baseline (probe trained on that model's own
activations). Compatible with results from both cross_model_probe_pipeline.py
and cross_model_original_probe_pipeline.py.

Two modes (--outer_grouping):
  train_datasets — x-axis = train groups,  one figure per (activation_model, layer)
  layers         — x-axis = layers,        one figure per (activation_model, train_dataset)

Within each outer group the bars are clustered by probe type (inner grouping).
Within each probe-type cluster, one bar per source model (color-coded).

Missing result files or train keys are skipped with a warning.

Figures written to:
  train_datasets mode: {figures_dir}/{activation_model}/layer_{layer}/transfer_summary.png
  layers mode:         {figures_dir}/{activation_model}/{train_dataset}/transfer_summary.png

Config format:
  {
      "outer_grouping": "train_datasets",
      "layers": [8, 12, 15],
      "train_datasets": ["cities", "larger_than", ...],
      "activation_datasets": ["cities", "neg_cities", ...],
      "figures": [
          {
              "activation_model": "llama-3.1-8b-instruct",
              "probe_sources": [
                  {"probe_model": "llama-3.1-8b-instruct", "label": "instruct",   "in_domain": true},
                  {"probe_model": "llama-3.1-8b",          "label": "base → instruct"},
                  {"probe_model": "llama-3.1-8b-instruct-bad-medical-advice",
                   "label": "bad-medical → instruct"}
              ]
          }
      ]
  }

  in_domain: true  — load from {probes_dir}/{activation_model}/layer_{layer}/results.json
  in_domain: false — load from {cross_model_probes_dir}/{probe_model}__{activation_model}/layer_{layer}/results.json

Example:
    python scripts/visualize_transfer.py \\
        --config data/configs/probes/transfer_config.json \\
        --probes_dir /scratch/geometry_of_truth/probes \\
        --cross_model_probes_dir /scratch/geometry_of_truth/cross_model_probes
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MODEL_REGISTRY
from src.plot_utils import (
    PROBE_DISPLAY,
    SOURCE_PALETTE,
    detect_probe_cols,
    load_probe_scores,
    plot_transfer_summary,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_source_df(
    source: dict,
    activation_model: str,
    layer: int,
    train_key: str,
    probes_dir: str,
    cross_model_probes_dir: str,
    activation_datasets: list[str],
) -> "pd.DataFrame | None":
    """
    Load per-test-dataset score DataFrame for one probe source at one
    (layer, train_key). Returns None and prints a warning if data is missing.
    """
    probe_model = source["probe_model"]
    label       = source["label"]
    in_domain   = source.get("in_domain", False)

    if in_domain:
        results_path = os.path.join(
            probes_dir, activation_model, f"layer_{layer}", "results.json"
        )
    else:
        pair_key     = f"{probe_model}__{activation_model}"
        results_path = os.path.join(
            cross_model_probes_dir, pair_key, f"layer_{layer}", "results.json"
        )

    if not os.path.exists(results_path):
        print(f"    WARNING: {results_path} not found — skipping '{label}'")
        return None

    full_df = load_probe_scores(results_path)
    present = set(full_df.index.get_level_values("train_group"))

    if train_key not in present:
        print(f"    WARNING: train_group '{train_key}' not in {results_path} — skipping '{label}'")
        return None

    df        = full_df.loc[train_key]
    available = [ds for ds in activation_datasets if ds in df.index]

    if not available:
        print(f"    WARNING: none of the activation_datasets found in {results_path} — skipping '{label}'")
        return None

    return df.loc[available]


def build_data(
    fig_cfg: dict,
    outer_labels: list[str],
    layer_for_outer,        # int if outer=train_datasets; callable str->int if outer=layers
    train_key_for_outer,    # callable str->str if outer=layers; str if outer=train_datasets
    probes_dir: str,
    cross_model_probes_dir: str,
    activation_datasets: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    activation_model = fig_cfg["activation_model"]
    data = {}

    for outer in outer_labels:
        layer     = layer_for_outer(outer) if callable(layer_for_outer) else layer_for_outer
        train_key = train_key_for_outer(outer) if callable(train_key_for_outer) else train_key_for_outer
        data[outer] = {}

        for source in fig_cfg["probe_sources"]:
            df = load_source_df(
                source, activation_model, layer, train_key,
                probes_dir, cross_model_probes_dir, activation_datasets,
            )
            if df is not None:
                data[outer][source["label"]] = df

    return data


def first_df(data: dict) -> "pd.DataFrame | None":
    """Return the first available DataFrame in a nested dict, or None."""
    for inner in data.values():
        for df in inner.values():
            return df
    return None


def run(cfg: dict, probes_dir: str, cross_model_probes_dir: str, figures_dir: str) -> None:
    outer_grouping      = cfg.get("outer_grouping", "train_datasets")
    layers              = cfg["layers"]
    train_datasets      = cfg["train_datasets"]
    activation_datasets = cfg["activation_datasets"]

    for fig_cfg in cfg["figures"]:
        activation_model = fig_cfg["activation_model"]
        source_labels    = [s["label"] for s in fig_cfg["probe_sources"]]
        source_colors    = {
            s["label"]: SOURCE_PALETTE[j % len(SOURCE_PALETTE)]
            for j, s in enumerate(fig_cfg["probe_sources"])
        }
        act_display = (
            MODEL_REGISTRY[activation_model].display_name
            if activation_model in MODEL_REGISTRY
            else activation_model
        )

        print(f"\n{'=' * 70}")
        print(f"Activation model: {act_display}  |  outer_grouping: {outer_grouping}")
        print(f"{'=' * 70}")

        if outer_grouping == "train_datasets":
            outer_labels = train_datasets
            for layer in layers:
                print(f"\n  Layer {layer}")
                data = build_data(
                    fig_cfg,
                    outer_labels=outer_labels,
                    layer_for_outer=layer,
                    train_key_for_outer=lambda x: x,
                    probes_dir=probes_dir,
                    cross_model_probes_dir=cross_model_probes_dir,
                    activation_datasets=activation_datasets,
                )
                sample = first_df(data)
                if sample is None:
                    print(f"  WARNING: no data found — skipping layer {layer}")
                    continue

                fig_dir = os.path.join(figures_dir, activation_model, f"layer_{layer}")
                os.makedirs(fig_dir, exist_ok=True)

                for probe in detect_probe_cols(sample):
                    probe_label = PROBE_DISPLAY.get(probe, probe)
                    title       = f"Probe Transfer [{probe_label}] → {act_display}  |  layer {layer}"
                    fig         = plot_transfer_summary(data, [probe], source_labels, source_colors, title=title)
                    out_path    = os.path.join(fig_dir, f"{probe}_transfer_summary.png")
                    fig.write_image(out_path)
                    print(f"  Saved: {out_path}")

        else:  # outer_grouping == "layers"
            outer_labels = [f"layer {l}" for l in layers]
            layer_map    = {f"layer {l}": l for l in layers}

            for train_key in train_datasets:
                print(f"\n  Train group: {train_key}")
                data = build_data(
                    fig_cfg,
                    outer_labels=outer_labels,
                    layer_for_outer=lambda x: layer_map[x],
                    train_key_for_outer=train_key,
                    probes_dir=probes_dir,
                    cross_model_probes_dir=cross_model_probes_dir,
                    activation_datasets=activation_datasets,
                )
                sample = first_df(data)
                if sample is None:
                    print(f"  WARNING: no data found — skipping train_group '{train_key}'")
                    continue

                safe_key = train_key.replace("+", "_")
                fig_dir  = os.path.join(figures_dir, activation_model, safe_key)
                os.makedirs(fig_dir, exist_ok=True)

                for probe in detect_probe_cols(sample):
                    probe_label = PROBE_DISPLAY.get(probe, probe)
                    title       = f"Probe Transfer [{probe_label}] → {act_display}  |  train: {train_key}"
                    fig         = plot_transfer_summary(data, [probe], source_labels, source_colors, title=title)
                    out_path    = os.path.join(fig_dir, f"{probe}_transfer_summary.png")
                    fig.write_image(out_path)
                    print(f"  Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-model probe transfer visualization")
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument(
        "--probes_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes",
        help="Root dir for in-domain probe results",
    )
    parser.add_argument(
        "--cross_model_probes_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/cross_model_probes",
        help="Root dir for cross-model probe results",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures/transfer",
        help="Root dir for output PNG figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    run(cfg, args.probes_dir, args.cross_model_probes_dir, args.figures_dir)


if __name__ == "__main__":
    main()
