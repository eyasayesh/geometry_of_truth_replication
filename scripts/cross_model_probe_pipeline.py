#!/usr/bin/env python3
"""
Cross-model probe transfer pipeline.

Loads probes trained on one model's activations and evaluates them on another
model's activations at the same layer. Models must share the same hidden
dimension; pairs with mismatched dimensions are skipped with a warning.

Results are saved to:
  {output_dir}/{probe_model}__{activation_model}/layer_{layer}/results.json

Config format:
  {
      "probe_datasets": ["cities", "larger_than", "cities+neg_cities"],
      "activation_datasets": ["cities", "sp_en_trans", "neg_cities"],
      "pairs": [
          {
              "probe_model": "llama-3.1-8b",
              "activation_model": "llama-3.1-8b-instruct",
              "layers": [8, 12, 15, 18]
          }
      ]
  }

  probe_datasets entries are train_key strings that match the probe directory
  names created by probe_pipeline.py:
    single dataset:   "cities"
    combined dataset: "cities+neg_cities"  (joined with "+" during training)

Example:
    python scripts/cross_model_probe_pipeline.py \\
        --config data/configs/probes/cross_model_probe_config.json
"""

import argparse
import json
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations import load_acts
from src.data import load_dataset, ALL_DATASETS
from src.models import MODEL_REGISTRY
from src.probes import LRProbe, MassMeanProbe


def load_acts_and_labels(
    model_name: str,
    dataset_name: str,
    layer: int,
    acts_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    acts = load_acts(model_name, dataset_name, layer, output_dir=acts_dir, center=True)
    labels_path = os.path.join(acts_dir, model_name, dataset_name, "labels.pt")
    if os.path.exists(labels_path):
        labels = torch.load(labels_path, weights_only=True).long()
    else:
        _, raw_labels = load_dataset(dataset_name)
        labels = torch.tensor(raw_labels, dtype=torch.long)
    n = min(len(acts), len(labels))
    return acts[:n], labels[:n]


def load_probes(
    probe_model_dir: str,
    train_key: str,
    layer: int,
) -> tuple["LRProbe | None", "MassMeanProbe | None", "MassMeanProbe | None"]:
    """
    Load all three probe types for a given model dir / layer / train_key.
    Returns (None, None, None) if any probe file is missing.
    """
    base = os.path.join(probe_model_dir, f"layer_{layer}", "probes", train_key)
    lr_path     = os.path.join(base, "lr.pt")
    mm_path     = os.path.join(base, "mm.pt")
    mm_cov_path = os.path.join(base, "mm_cov.pt")

    if not all(os.path.exists(p) for p in (lr_path, mm_path, mm_cov_path)):
        return None, None, None

    return (
        LRProbe.load(lr_path),
        MassMeanProbe.load(mm_path),
        MassMeanProbe.load(mm_cov_path),
    )


def run_pair(
    probe_model: str,
    activation_model: str,
    layers: list[int],
    probe_datasets: list[str],
    activation_datasets: list[str],
    probes_dir: str,
    acts_dir: str,
    output_dir: str,
) -> None:
    """
    Evaluate probes trained on probe_model against activations from activation_model.
    Skips with a warning if the two models have different hidden dimensions.
    """
    probe_cfg = MODEL_REGISTRY[probe_model]
    act_cfg   = MODEL_REGISTRY[activation_model]

    if probe_cfg.hidden_size != act_cfg.hidden_size:
        warnings.warn(
            f"Skipping pair ({probe_model!r} → {activation_model!r}): "
            f"hidden_size mismatch ({probe_cfg.hidden_size} vs {act_cfg.hidden_size}).",
            stacklevel=2,
        )
        return

    pair_key        = f"{probe_model}__{activation_model}"
    probe_model_dir = os.path.join(probes_dir, probe_model)

    print(f"\n{'=' * 70}")
    print(f"Pair: {probe_cfg.display_name}  →  {act_cfg.display_name}")
    print(f"  hidden_size = {probe_cfg.hidden_size}")
    print(f"  layers      = {layers}")
    print(f"{'=' * 70}")

    for layer in layers:
        print(f"\n  Layer {layer}")
        results = {}

        for train_key in probe_datasets:
            lr_probe, mm_probe, mm_cov_probe = load_probes(probe_model_dir, train_key, layer)
            if lr_probe is None:
                print(
                    f"    WARNING: No saved probes at "
                    f"'{probe_model_dir}/layer_{layer}/probes/{train_key}' — skipping."
                )
                continue

            results[train_key] = {
                "layer":            layer,
                "probe_model":      probe_model,
                "activation_model": activation_model,
                "scores":           {},
            }

            for act_ds in activation_datasets:
                try:
                    acts, labels = load_acts_and_labels(activation_model, act_ds, layer, acts_dir)
                except FileNotFoundError as e:
                    print(f"    WARNING: {e} — skipping {act_ds}.")
                    continue

                scores = {
                    "lr":     round(lr_probe.score(acts, labels), 4),
                    "mm":     round(mm_probe.score(acts, labels), 4),
                    "mm_cov": round(mm_cov_probe.score(acts, labels), 4),
                    "n":      len(acts),
                }
                results[train_key]["scores"][act_ds] = scores
                print(
                    f"    probe={train_key:30s}  act_ds={act_ds:30s}  "
                    f"LR={scores['lr']:.3f}  MM={scores['mm']:.3f}  MM+cov={scores['mm_cov']:.3f}"
                )

        results_path = os.path.join(output_dir, pair_key, f"layer_{layer}", "results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  → Saved: {results_path}")


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)

    for i, pair in enumerate(cfg.get("pairs", [])):
        for key in ("probe_model", "activation_model"):
            if pair[key] not in MODEL_REGISTRY:
                raise ValueError(f"pairs[{i}].{key}: unknown model {pair[key]!r}")

    unknown_act_ds = [ds for ds in cfg.get("activation_datasets", []) if ds not in ALL_DATASETS]
    if unknown_act_ds:
        raise ValueError(f"Unknown activation_dataset(s): {unknown_act_ds}")

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-model probe transfer evaluation")
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument(
        "--probes_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes",
        help=(
            "Root directory containing saved probes. "
            "Expected layout: {probes_dir}/{model}/layer_{layer}/probes/{train_key}/*.pt"
        ),
    )
    parser.add_argument(
        "--acts_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts",
        help="Root directory containing cached activations",
    )
    parser.add_argument(
        "--output_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/cross_model_probes",
        help=(
            "Root directory for cross-model probe results. "
            "Results are written to: {output_dir}/{probe_model}__{activation_model}/layer_{layer}/results.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    probe_datasets      = cfg["probe_datasets"]
    activation_datasets = cfg["activation_datasets"]

    for pair in cfg["pairs"]:
        run_pair(
            probe_model=pair["probe_model"],
            activation_model=pair["activation_model"],
            layers=pair["layers"],
            probe_datasets=probe_datasets,
            activation_datasets=activation_datasets,
            probes_dir=args.probes_dir,
            acts_dir=args.acts_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
