#!/usr/bin/env python3
"""
Probe training and cross-dataset generalization pipeline (Section 5 of paper).

For each model and layer, trains LR and MassMean probes on one or more datasets
combined, then evaluates accuracy on all test datasets.

Each training configuration is a tuple of dataset names. Single-dataset tuples
replicate the original paper setup; multi-dataset tuples train on the combined
activations of all listed datasets.

Results are written to:
  {output_dir}/{model}/layer_{layer}/probes/{train_key}/{probe_type}.pt
  {output_dir}/{model}/layer_{layer}/results.json

Config file format:
  {
      "seed": 42,
      "test_split": 0.2,
      "models": {
          "llama-3.1-8b-instruct": { "layers": [12, 14, 16, 18, 20] }
      },
      "train_groups": [
          ["cities"],
          ["sp_en_trans"],
          ["cities", "sp_en_trans"]
      ],
      "test_datasets": ["cities", "neg_cities", ...]
  }

Example:
    # Run from a config file (iterates all models and layers)
    python scripts/probe_pipeline.py --config data/configs/probes/probe_config.json

    # Single model/layer with CLI flags
    python scripts/probe_pipeline.py \\
        --model llama-3.1-8b-instruct \\
        --layer 16 \\
        --train_groups cities sp_en_trans "cities,sp_en_trans"
"""

import argparse
import json
import os
import sys

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
    # Align lengths (activations may be subsampled)
    n = min(len(acts), len(labels))
    return acts[:n], labels[:n]


def load_group_acts_and_labels(
    model_name: str,
    group: tuple[str, ...],
    layer: int,
    acts_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and concatenate activations/labels for all datasets in a group."""
    all_acts, all_labels = [], []
    for ds in group:
        a, l = load_acts_and_labels(model_name, ds, layer, acts_dir)
        all_acts.append(a)
        all_labels.append(l)
    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)


def train_and_eval(
    model_name: str,
    layer: int,
    train_groups: list[tuple[str, ...]],
    test_datasets: list[str],
    acts_dir: str,
    output_dir: str,
    test_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    For each training group (tuple of one or more datasets), fit LR and MM probes
    on the combined activations, then evaluate on all test datasets.
    Returns nested results dict keyed by the group name (datasets joined with '+').
    """
    torch.manual_seed(seed)
    results = {}

    for group in train_groups:
        train_key = "+".join(group)
        print(f"  Train group: {train_key}")
        try:
            acts, labels = load_group_acts_and_labels(model_name, group, layer, acts_dir)
        except FileNotFoundError as e:
            print(f"    WARNING: {e} — skipping.")
            continue

        # Train/test split on the combined group data
        n = len(acts)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        n_test = max(1, int(n * test_split))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        train_acts, train_labels = acts[train_idx], labels[train_idx]
        in_domain_acts, in_domain_labels = acts[test_idx], labels[test_idx]

        # Fit probes
        lr_probe = LRProbe().fit(train_acts, train_labels)
        mm_probe = MassMeanProbe().fit(train_acts, train_labels, use_cov=False)
        mm_cov_probe = MassMeanProbe().fit(train_acts, train_labels, use_cov=True)

        # Save probes
        probe_dir = os.path.join(output_dir, model_name, f"layer_{layer}", "probes", train_key)
        lr_probe.save(os.path.join(probe_dir, "lr.pt"))
        mm_probe.save(os.path.join(probe_dir, "mm.pt"))
        mm_cov_probe.save(os.path.join(probe_dir, "mm_cov.pt"))

        results[train_key] = {"layer": layer, "train_datasets": list(group), "n_train": len(train_idx), "scores": {}}

        # Evaluate on all test datasets
        for test_ds in test_datasets:
            try:
                # Use the held-out split when the test dataset is the sole member of a single-dataset group
                if group == (test_ds,):
                    t_acts, t_labels = in_domain_acts, in_domain_labels
                else:
                    t_acts, t_labels = load_acts_and_labels(model_name, test_ds, layer, acts_dir)
            except FileNotFoundError:
                continue

            results[train_key]["scores"][test_ds] = {
                "lr":     round(lr_probe.score(t_acts, t_labels), 4),
                "mm":     round(mm_probe.score(t_acts, t_labels), 4),
                "mm_cov": round(mm_cov_probe.score(t_acts, t_labels), 4),
                "n":      len(t_acts),
            }
            print(
                f"    → {test_ds:35s}  "
                f"LR={results[train_key]['scores'][test_ds]['lr']:.3f}  "
                f"MM={results[train_key]['scores'][test_ds]['mm']:.3f}  "
                f"MM+cov={results[train_key]['scores'][test_ds]['mm_cov']:.3f}"
            )

    return results


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)

    unknown = [m for m in cfg["models"] if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s) in config: {unknown}")

    unknown_ds = [
        ds
        for group in cfg.get("train_groups", [])
        for ds in group
        if ds not in ALL_DATASETS
    ] + [ds for ds in cfg.get("test_datasets", []) if ds not in ALL_DATASETS]
    if unknown_ds:
        raise ValueError(f"Unknown dataset(s) in config: {unknown_ds}")

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Probe training and cross-dataset generalization")
    parser.add_argument(
        "--config", default=None,
        help="JSON config file (iterates all models and layers; overrides --model/--layer/--train_groups/--test_datasets)",
    )
    parser.add_argument("--model", default=None, help="Model name from MODEL_REGISTRY")
    parser.add_argument("--layer", type=int, default=None, help="Layer to probe")
    parser.add_argument(
        "--train_groups", nargs="+", default=None,
        help=(
            "Training groups: each is a comma-separated list of datasets. "
            "E.g. cities 'cities,sp_en_trans' trains two probes: one on cities "
            "alone and one on both combined. Default: each dataset individually."
        ),
    )
    parser.add_argument(
        "--test_datasets", nargs="+", default=None,
        help="Datasets to evaluate on (default: all)",
    )
    parser.add_argument(
        "--acts_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts",
    )
    parser.add_argument(
        "--output_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes",
    )
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_one(model_name, layer, train_groups, test_datasets, acts_dir, output_dir, test_split, seed):
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_REGISTRY[model_name].display_name}  |  layer: {layer}")
    print(f"{'=' * 60}")
    print(f"Train groups:  {['+'.join(g) for g in train_groups]}")
    print(f"Test datasets: {test_datasets}")
    print()

    results = train_and_eval(
        model_name=model_name,
        layer=layer,
        train_groups=train_groups,
        test_datasets=test_datasets,
        acts_dir=acts_dir,
        output_dir=output_dir,
        test_split=test_split,
        seed=seed,
    )

    results_path = os.path.join(output_dir, model_name, f"layer_{layer}", "results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}\n")


def main():
    args = parse_args()

    if args.config is not None:
        cfg = load_config(args.config)
        train_groups  = [tuple(g) for g in cfg.get("train_groups", [[ds] for ds in ALL_DATASETS])]
        test_datasets = cfg.get("test_datasets", ALL_DATASETS)
        test_split    = cfg.get("test_split", args.test_split)
        seed          = cfg.get("seed", args.seed)

        for model_name, model_cfg in cfg["models"].items():
            for layer in model_cfg["layers"]:
                run_one(model_name, layer, train_groups, test_datasets,
                        args.acts_dir, args.output_dir, test_split, seed)
    else:
        if args.model is None or args.layer is None:
            print("ERROR: provide either --config or both --model and --layer.")
            sys.exit(1)
        if args.model not in MODEL_REGISTRY:
            print(f"Unknown model '{args.model}'. Available: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

        train_groups  = [tuple(g.split(",")) for g in args.train_groups] if args.train_groups else [(ds,) for ds in ALL_DATASETS]
        test_datasets = args.test_datasets or ALL_DATASETS

        run_one(args.model, args.layer, train_groups, test_datasets,
                args.acts_dir, args.output_dir, args.test_split, args.seed)


if __name__ == "__main__":
    main()
