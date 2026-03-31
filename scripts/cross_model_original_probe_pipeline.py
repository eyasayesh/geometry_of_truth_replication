#!/usr/bin/env python3
"""
Cross-model probe transfer pipeline for original geometry-of-truth probes
(LRProbe, MMProbe).

Loads probes trained by original_probe_pipeline.py on one model's activations
and evaluates them on another model's activations at the same layer. Models
must share the same hidden dimension; mismatched pairs are skipped with a
warning.

Probe files expected at:
  {probes_dir}/{probe_model}/layer_{layer}/probes/{train_key}/lr.pt
  {probes_dir}/{probe_model}/layer_{layer}/probes/{train_key}/mm.pt

Results are saved to:
  {output_dir}/{probe_model}__{activation_model}/layer_{layer}/results.json

Result scores per test dataset: lr, mm, mm_iid  (CCS excluded).

Uses the same config format as cross_model_probe_pipeline.py.

Example:
    python scripts/cross_model_original_probe_pipeline.py \\
        --config data/configs/probes/cross_model_probe_config.json \\
        --probes_dir /scratch/geometry_of_truth/probes_original \\
        --output_dir /scratch/geometry_of_truth/cross_model_probes_original
"""

import argparse
import json
import os
import sys
import warnings

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_DIR  = os.path.join(REPO_ROOT, "geometry-of-truth")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ORIG_DIR)

from probes import LRProbe, MMProbe          # original implementations
from src.activations import load_acts
from src.data import load_dataset, ALL_DATASETS
from src.models import MODEL_REGISTRY


# ── Probe loading ─────────────────────────────────────────────────────────────

def load_lr_probe(path: str, d_in: int) -> LRProbe:
    probe = LRProbe(d_in)
    probe.load_state_dict(torch.load(path, weights_only=True))
    probe.eval()
    return probe


def load_mm_probe(path: str) -> MMProbe:
    state   = torch.load(path, weights_only=True)
    # Reconstruct directly from saved direction and inv parameters
    probe   = MMProbe(state["direction"], inv=state["inv"])
    probe.eval()
    return probe


def load_probes(
    probe_model_dir: str,
    train_key: str,
    layer: int,
    d_in: int,
) -> tuple["LRProbe | None", "MMProbe | None"]:
    """
    Load LRProbe and MMProbe for a given model dir / layer / train_key.
    Returns (None, None) if either probe file is missing.
    """
    base    = os.path.join(probe_model_dir, f"layer_{layer}", "probes", train_key)
    lr_path = os.path.join(base, "lr.pt")
    mm_path = os.path.join(base, "mm.pt")

    if not all(os.path.exists(p) for p in (lr_path, mm_path)):
        return None, None

    return load_lr_probe(lr_path, d_in), load_mm_probe(mm_path)


# ── Activation / label loading ────────────────────────────────────────────────

def load_acts_and_labels(
    model_name: str,
    dataset_name: str,
    layer: int,
    acts_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    acts = load_acts(model_name, dataset_name, layer, output_dir=acts_dir, center=True)
    labels_path = os.path.join(acts_dir, model_name, dataset_name, "labels.pt")
    if os.path.exists(labels_path):
        labels = torch.load(labels_path, weights_only=True).float()
    else:
        _, raw_labels = load_dataset(dataset_name)
        labels = torch.tensor(raw_labels, dtype=torch.float32)
    n = min(len(acts), len(labels))
    return acts[:n], labels[:n]


# ── Accuracy ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def accuracy(probe, acts: torch.Tensor, labels: torch.Tensor, iid: bool = False) -> float:
    preds = probe.pred(acts, iid=iid) if isinstance(probe, MMProbe) else probe.pred(acts)
    return float((preds == labels).float().mean())


# ── Main pipeline ─────────────────────────────────────────────────────────────

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
    d_in            = probe_cfg.hidden_size

    print(f"\n{'=' * 70}")
    print(f"Pair: {probe_cfg.display_name}  →  {act_cfg.display_name}")
    print(f"  hidden_size = {d_in}")
    print(f"  layers      = {layers}")
    print(f"{'=' * 70}")

    for layer in layers:
        print(f"\n  Layer {layer}")
        results = {}

        for train_key in probe_datasets:
            lr_probe, mm_probe = load_probes(probe_model_dir, train_key, layer, d_in)
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
                    "lr":     round(accuracy(lr_probe, acts, labels), 4),
                    "mm":     round(accuracy(mm_probe, acts, labels, iid=False), 4),
                    "mm_iid": round(accuracy(mm_probe, acts, labels, iid=True), 4),
                    "n":      len(acts),
                }
                results[train_key]["scores"][act_ds] = scores
                print(
                    f"    probe={train_key:30s}  act_ds={act_ds:30s}  "
                    f"LR={scores['lr']:.3f}  MM={scores['mm']:.3f}  MM+iid={scores['mm_iid']:.3f}"
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
    parser = argparse.ArgumentParser(
        description="Cross-model probe transfer evaluation (original geometry-of-truth probes)"
    )
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument(
        "--probes_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes_original",
        help="Root directory containing saved original probes",
    )
    parser.add_argument(
        "--acts_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts",
        help="Root directory containing cached activations",
    )
    parser.add_argument(
        "--output_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/cross_model_probes_original",
        help="Root directory for cross-model probe results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    for pair in cfg["pairs"]:
        run_pair(
            probe_model=pair["probe_model"],
            activation_model=pair["activation_model"],
            layers=pair["layers"],
            probe_datasets=cfg["probe_datasets"],
            activation_datasets=cfg["activation_datasets"],
            probes_dir=args.probes_dir,
            acts_dir=args.acts_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
