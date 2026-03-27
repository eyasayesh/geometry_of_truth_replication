#!/usr/bin/env python3
"""
Visualize truth representations via PCA.

Loads saved activations for a model/dataset at a given layer, runs PCA,
prints explained variance for the top k PCs, and saves a 2D scatter plot.

Activation extraction (if not already done):
    python scripts/generate_acts.py \
        --model llama-3.2-1b \
        --datasets cities \
        --layers 8 \
        --output_dir /path/to/scratch/acts

PCA visualization:
    python scripts/visualize_pca.py \
        --model llama-3.2-1b \
        --dataset cities \
        --layer 8 \
        --acts_dir /path/to/scratch/acts \
        --output_dir /path/to/scratch/figures
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations import load_acts
from src.data import load_dataset, ALL_DATASETS
from src.models import MODEL_REGISTRY
from src.pca import run_pca, plot_pca


def parse_args():
    parser = argparse.ArgumentParser(description="PCA visualization of truth representations")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--layers", required=True, nargs="+", type=int, help="Layer index/indices to visualize")
    parser.add_argument(
        "--acts_dir",
        required=True,
        help="Root directory where activations are saved (e.g. /scratch/acts)",
    )
    parser.add_argument(
        "--output_dir",
        default="figures/pca",
        help="Directory to save PNG figures (default: figures/pca)",
    )
    parser.add_argument(
        "--pca_output_dir",
        default=None,
        help="Directory to save PCA results as .pt files (components, projections, explained variance). If not set, PCA results are not saved.",
    )
    parser.add_argument(
        "--n_pcs",
        type=int,
        default=10,
        help="Number of PCs to report explained variance for (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.pca_output_dir:
        os.makedirs(args.pca_output_dir, exist_ok=True)

    _, labels = load_dataset(args.dataset)
    config = MODEL_REGISTRY[args.model]

    for layer in args.layers:
        print(f"\n--- Layer {layer} ---")
        try:
            acts = load_acts(
                model_name=args.model,
                dataset_name=args.dataset,
                layer=layer,
                output_dir=args.acts_dir,
                center=False,  # PCA will center + scale internally
            )
        except FileNotFoundError:
            print(f"  WARNING: no activations found for layer {layer}, skipping.")
            continue
        print(f"  Shape: {tuple(acts.shape)}")

        k = min(args.n_pcs, acts.shape[1], acts.shape[0] - 1)
        print(f"  Running PCA (top {k} components)...")
        result = run_pca(acts, k=k)

        print("  Explained variance:")
        cumulative = 0.0
        for i, ev in enumerate(result.explained_var_ratio):
            cumulative += ev
            print(f"    PC{i+1}: {ev:.2%}  (cumulative: {cumulative:.2%})")

        title = f"{config.display_name} — {args.dataset} — layer {layer}<br>"
        title += f"<sup>PC1: {result.explained_var_ratio[0]:.1%}, PC2: {result.explained_var_ratio[1]:.1%} explained variance</sup>"
        fig = plot_pca(result, labels, title=title)

        out_path = os.path.join(args.output_dir, f"pca_{args.model}_{args.dataset}_layer{layer}.png")
        fig.write_image(out_path)
        print(f"  Saved figure: {out_path}")

        if args.pca_output_dir:
            pca_path = os.path.join(
                args.pca_output_dir,
                f"pca_{args.model}_{args.dataset}_layer{layer}.pt",
            )
            torch.save({
                "components": result.components,
                "projections": result.projections,
                "explained_var_ratio": result.explained_var_ratio,
                "model": args.model,
                "dataset": args.dataset,
                "layer": layer,
            }, pca_path)
            print(f"  Saved PCA: {pca_path}")


if __name__ == "__main__":
    main()
