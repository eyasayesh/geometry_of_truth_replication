#!/usr/bin/env python3
"""
Probe training pipeline using the original geometry-of-truth probe implementations
(LRProbe, MMProbe, CCSProbe) for comparison against our sklearn/MassMean probes.

Trains on activations extracted by original_pca_pipeline.py (acts_original dir).
Results written to: {output_dir}/{model_name}/layer_{layer}/results.json
Probes written to:  {output_dir}/{model_name}/layer_{layer}/probes/{train_key}/{probe_type}.pt

Differences from our probe_pipeline.py:
  LRProbe  — PyTorch AdamW, no bias, direction not unit-normalized
  MMProbe  — full (non-pooled) covariance via pinv, iid=True applies correction
  CCSProbe — contrastive consistency probe, requires paired pos/neg datasets

Example:
    python scripts/original_probe_pipeline.py \\
        --model_name llama-2-13b \\
        --layers 12 16 20 \\
        --train_groups cities sp_en_trans

    python scripts/original_probe_pipeline.py \\
        --model_name llama-2-13b \\
        --layers 16 \\
        --train_groups cities \\
        --test_datasets cities neg_cities sp_en_trans
"""

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_DIR  = os.path.join(REPO_ROOT, "geometry-of-truth")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ORIG_DIR)

from probes import LRProbe, MMProbe, CCSProbe  # original implementations
from src.data import load_dataset, ALL_DATASETS

# Default pairing of positive datasets to their negations for CCSProbe
NEG_PAIRS: dict[str, str] = {
    "cities":       "neg_cities",
    "sp_en_trans":  "neg_sp_en_trans",
    "neg_cities":       "cities",
    "neg_sp_en_trans":  "sp_en_trans",
}


# ── Activation loading ────────────────────────────────────────────────────────

def load_acts(
    model_name: str,
    dataset_name: str,
    layer: int,
    acts_dir: str,
    center: bool = True,
) -> torch.Tensor:
    from glob import glob
    directory = os.path.join(acts_dir, model_name, dataset_name)
    files = sorted(
        glob(os.path.join(directory, f"layer_{layer}_*.pt")),
        key=lambda f: int(os.path.basename(f).split("_")[-1].replace(".pt", "")),
    )
    if not files:
        raise FileNotFoundError(
            f"No activations at '{directory}' for layer {layer}. "
            "Run original_pca_pipeline.py first."
        )
    acts = torch.cat(
        [torch.load(f, weights_only=True) for f in files], dim=0
    ).detach().float()
    if center:
        acts = acts - acts.mean(dim=0)
    return acts


def load_labels(
    model_name: str,
    dataset_name: str,
    acts_dir: str,
) -> torch.Tensor:
    labels_path = os.path.join(acts_dir, model_name, dataset_name, "labels.pt")
    if os.path.exists(labels_path):
        return torch.load(labels_path, weights_only=True).float()
    _, raw = load_dataset(dataset_name)
    return torch.tensor(raw, dtype=torch.float32)


def load_acts_and_labels(
    model_name: str,
    dataset_name: str,
    layer: int,
    acts_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    acts   = load_acts(model_name, dataset_name, layer, acts_dir)
    labels = load_labels(model_name, dataset_name, acts_dir)
    n = min(len(acts), len(labels))
    return acts[:n], labels[:n]


def load_group(
    model_name: str,
    group: tuple[str, ...],
    layer: int,
    acts_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_acts, all_labels = [], []
    for ds in group:
        a, l = load_acts_and_labels(model_name, ds, layer, acts_dir)
        all_acts.append(a)
        all_labels.append(l)
    return torch.cat(all_acts), torch.cat(all_labels)


# ── Probe accuracy ────────────────────────────────────────────────────────────

def accuracy(probe, acts: torch.Tensor, labels: torch.Tensor, iid: bool = False) -> float:
    """Compute classification accuracy. iid flag only affects MMProbe."""
    preds = probe.pred(acts, iid=iid) if isinstance(probe, (MMProbe, CCSProbe)) else probe.pred(acts)
    return float((preds == labels).float().mean())


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(
    model_name: str,
    layer: int,
    train_groups: list[tuple[str, ...]],
    test_datasets: list[str],
    acts_dir: str,
    output_dir: str,
    test_split: float = 0.2,
    seed: int = 42,
    lr_epochs: int = 1000,
    lr_lr: float = 0.001,
    lr_wd: float = 0.1,
) -> dict:
    torch.manual_seed(seed)
    results = {}

    for group in train_groups:
        train_key = "+".join(group)
        print(f"  Train group: {train_key}")

        try:
            acts, labels = load_group(model_name, group, layer, acts_dir)
        except FileNotFoundError as e:
            print(f"    WARNING: {e} — skipping.")
            continue

        # Train/test split
        n = len(acts)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        n_test = max(1, int(n * test_split))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        train_acts,   train_labels   = acts[train_idx],  labels[train_idx]
        indomain_acts, indomain_labels = acts[test_idx], labels[test_idx]

        # ── LRProbe (PyTorch, no bias) ────────────────────────────────────────
        lr_probe = LRProbe.from_data(
            train_acts, train_labels,
            lr=lr_lr, weight_decay=lr_wd, epochs=lr_epochs,
        )

        # ── MMProbe (full covariance via pinv) ────────────────────────────────
        mm_probe = MMProbe.from_data(train_acts, train_labels)

        # ── CCSProbe (contrastive, requires paired neg dataset) ───────────────
        ccs_probe = None
        if len(group) == 1 and group[0] in NEG_PAIRS:
            neg_ds = NEG_PAIRS[group[0]]
            try:
                neg_acts, _ = load_acts_and_labels(model_name, neg_ds, layer, acts_dir)
                neg_acts = neg_acts[train_idx[:len(train_acts)]]  # align length
                ccs_probe = CCSProbe.from_data(
                    train_acts, neg_acts, labels=train_labels,
                    lr=lr_lr, weight_decay=lr_wd, epochs=lr_epochs,
                )
            except FileNotFoundError:
                print(f"    NOTE: neg dataset '{neg_ds}' not found — skipping CCSProbe.")

        # ── Save probes ───────────────────────────────────────────────────────
        probe_dir = os.path.join(output_dir, model_name, f"layer_{layer}", "probes", train_key)
        os.makedirs(probe_dir, exist_ok=True)
        torch.save(lr_probe.state_dict(), os.path.join(probe_dir, "lr.pt"))
        torch.save(mm_probe.state_dict(), os.path.join(probe_dir, "mm.pt"))
        if ccs_probe is not None:
            torch.save(ccs_probe.state_dict(), os.path.join(probe_dir, "ccs.pt"))

        results[train_key] = {
            "layer": layer,
            "train_datasets": list(group),
            "n_train": len(train_idx),
            "scores": {},
        }

        # ── Evaluate ──────────────────────────────────────────────────────────
        for test_ds in test_datasets:
            try:
                if group == (test_ds,):
                    t_acts, t_labels = indomain_acts, indomain_labels
                else:
                    t_acts, t_labels = load_acts_and_labels(model_name, test_ds, layer, acts_dir)
            except FileNotFoundError:
                continue

            entry = {
                "lr":       round(accuracy(lr_probe, t_acts, t_labels), 4),
                "mm":       round(accuracy(mm_probe, t_acts, t_labels, iid=False), 4),
                "mm_iid":   round(accuracy(mm_probe, t_acts, t_labels, iid=True), 4),
                "n":        len(t_acts),
            }
            if ccs_probe is not None:
                entry["ccs"] = round(accuracy(ccs_probe, t_acts, t_labels), 4)

            results[train_key]["scores"][test_ds] = entry

            line = (
                f"    → {test_ds:35s}  "
                f"LR={entry['lr']:.3f}  "
                f"MM={entry['mm']:.3f}  "
                f"MM+iid={entry['mm_iid']:.3f}"
            )
            if "ccs" in entry:
                line += f"  CCS={entry['ccs']:.3f}"
            print(line)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe pipeline using original geometry-of-truth probes"
    )
    parser.add_argument(
        "--model_name", required=True,
        help="Short model name used for the acts directory (e.g. llama-2-13b-hf)",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, required=True,
        help="Layer indices to train probes at",
    )
    parser.add_argument(
        "--train_groups", nargs="+", default=None,
        help=(
            "Training groups: each is a comma-separated list of datasets. "
            "E.g. cities 'cities,sp_en_trans'. Default: each dataset individually."
        ),
    )
    parser.add_argument(
        "--test_datasets", nargs="+", default=None,
        help="Datasets to evaluate on (default: all available)",
    )
    parser.add_argument(
        "--acts_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts_original",
        help="Root directory for activations (from original_pca_pipeline.py)",
    )
    parser.add_argument(
        "--output_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes_original",
        help="Root directory for probe files and results",
    )
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_epochs", type=int, default=1000)
    parser.add_argument("--lr_lr", type=float, default=0.001)
    parser.add_argument("--lr_wd", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve train groups
    if args.train_groups:
        train_groups = [tuple(g.split(",")) for g in args.train_groups]
    else:
        # Default: each dataset individually (only those with activations)
        train_groups = [(ds,) for ds in ALL_DATASETS]

    # Resolve test datasets
    test_datasets = args.test_datasets or ALL_DATASETS

    print(f"Model:         {args.model_name}")
    print(f"Layers:        {args.layers}")
    print(f"Train groups:  {['+'.join(g) for g in train_groups]}")
    print(f"Test datasets: {test_datasets}")
    print(f"acts_dir:      {args.acts_dir}")
    print()

    for layer in args.layers:
        print(f"{'=' * 60}")
        print(f"Layer: {layer}")
        print(f"{'=' * 60}")

        results = train_and_eval(
            model_name=args.model_name,
            layer=layer,
            train_groups=train_groups,
            test_datasets=test_datasets,
            acts_dir=args.acts_dir,
            output_dir=args.output_dir,
            test_split=args.test_split,
            seed=args.seed,
            lr_epochs=args.lr_epochs,
            lr_lr=args.lr_lr,
            lr_wd=args.lr_wd,
        )

        results_path = os.path.join(
            args.output_dir, args.model_name, f"layer_{layer}", "results.json"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Results saved to {results_path}\n")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
