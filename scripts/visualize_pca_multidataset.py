#!/usr/bin/env python3
"""
PCA visualization across multiple datasets for a single model and layer.

Loads activations for each dataset, concatenates them, runs a single joint PCA,
then plots each dataset in a distinct color with true/false shown by symbol.
Saves an interactive HTML figure.

Example:
    python scripts/visualize_pca_multidataset.py \\
        --model gemma-2-9b \\
        --layer 16 \\
        --datasets cities neg_cities sp_en_trans larger_than \\
        --acts_dir /scratch/acts \\
        --output_dir figures/pca
"""

import argparse
import os
import sys

import pandas as pd
import plotly.express as px
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations import load_acts
from src.data import load_dataset, ALL_DATASETS, DEFAULT_DATASETS_DIR
from src.models import MODEL_REGISTRY
from src.pca import run_pca


# Plotly symbol map: True=circle, False=x
SYMBOL_MAP = {1: "circle", 0: "x"}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-dataset PCA visualization")
    parser.add_argument("--model",    required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--layer",    required=True, type=int)
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Datasets to include (default: all)",
    )
    parser.add_argument(
        "--acts_dir", required=True,
        help="Root directory where activations are saved",
    )
    parser.add_argument(
        "--output_dir", default="figures/pca",
        help="Directory to save the HTML figure (default: figures/pca)",
    )
    parser.add_argument(
        "--n_pcs", type=int, default=3,
        help="Number of PCs to compute (default: 3)",
    )
    return parser.parse_args()


def load_statements(dataset_name: str) -> list[str]:
    """Load raw statements from the dataset CSV."""
    path = os.path.join(DEFAULT_DATASETS_DIR, f"{dataset_name}.csv")
    return pd.read_csv(path)["statement"].tolist()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = MODEL_REGISTRY[args.model]

    # ── Load activations and metadata per dataset ─────────────────────────────
    all_acts   = []
    all_labels = []
    all_datasets_col = []
    all_statements   = []

    for dataset_name in args.datasets:
        try:
            acts = load_acts(
                model_name=args.model,
                dataset_name=dataset_name,
                layer=args.layer,
                output_dir=args.acts_dir,
                center=False,
            )
        except FileNotFoundError:
            print(f"WARNING: no activations for {dataset_name} layer {args.layer} — skipping.")
            continue

        # Check for saved (possibly subsampled) labels
        labels_path = os.path.join(args.acts_dir, args.model, dataset_name, "labels.pt")
        if os.path.exists(labels_path):
            labels = torch.load(labels_path, weights_only=True).tolist()
            statements = load_statements(dataset_name)
            # Subsampled — we don't have the original indices so use blank statements
            # for rows beyond the original CSV length; in practice labels.pt length == acts rows
            if len(statements) == len(labels):
                pass  # not subsampled, all good
            else:
                statements = [""] * len(labels)
        else:
            _, labels = load_dataset(dataset_name)
            statements = load_statements(dataset_name)

        n = acts.shape[0]
        all_acts.append(acts)
        all_labels.extend(labels[:n])
        all_datasets_col.extend([dataset_name] * n)
        all_statements.extend(statements[:n])

        print(f"  Loaded {dataset_name}: {n} rows")

    if not all_acts:
        print("No activations found — exiting.")
        sys.exit(1)

    # ── Joint PCA ─────────────────────────────────────────────────────────────
    combined = torch.cat(all_acts, dim=0)
    print(f"\nRunning PCA on {combined.shape[0]} total statements...")
    result = run_pca(combined, k=args.n_pcs)
    ev = result.explained_var_ratio
    print(f"  PC1: {ev[0]:.2%}  PC2: {ev[1]:.2%}" + (f"  PC3: {ev[2]:.2%}" if len(ev) > 2 else ""))

    proj = result.projections.cpu().numpy()

    # ── Build dataframe ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "PC1":       proj[:, 0],
        "PC2":       proj[:, 1],
        "dataset":   all_datasets_col,
        "truth":     ["True" if l == 1 else "False" for l in all_labels],
        "statement": all_statements,
    })
    if args.n_pcs >= 3:
        df["PC3"] = proj[:, 2]

    df["symbol"] = df["truth"].map({"True": "circle", "False": "x"})

    # ── 2D scatter ────────────────────────────────────────────────────────────
    title = (f"{config.display_name} — layer {args.layer} — "
             f"PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%}")
    fig2d = px.scatter(
        df,
        x="PC1", y="PC2",
        color="dataset",
        symbol="truth",
        symbol_map={"True": "circle", "False": "x"},
        hover_data={"statement": True, "truth": True,
                    "PC1": False, "PC2": False, "symbol": False},
        title=title,
        template="plotly_white",
        opacity=0.7,
    )
    fig2d.update_yaxes(scaleanchor="x", scaleratio=1)
    fig2d.update_traces(marker_size=5)
    fig2d.update_layout(
        xaxis_title=f"PC1 ({ev[0]:.1%} var)",
        yaxis_title=f"PC2 ({ev[1]:.1%} var)",
    )

    datasets_str = "_".join(args.datasets) if len(args.datasets) <= 3 else f"{len(args.datasets)}datasets"
    out2d = os.path.join(args.output_dir, f"pca_multi_{args.model}_layer{args.layer}_{datasets_str}.png")
    fig2d.write_image(out2d)
    print(f"\nSaved 2D figure: {out2d}")

    # ── 3D scatter ────────────────────────────────────────────────────────────
    if args.n_pcs >= 3:
        title3d = (f"{config.display_name} — layer {args.layer} — "
                   f"PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%}, PC3: {ev[2]:.1%}")
        fig3d = px.scatter_3d(
            df,
            x="PC1", y="PC2", z="PC3",
            color="dataset",
            symbol="truth",
            symbol_map={"True": "circle", "False": "x"},
            hover_data={"statement": True, "truth": True},
            title=title3d,
            template="plotly_white",
            opacity=0.7,
        )
        fig3d.update_traces(marker_size=3)
        out3d = os.path.join(args.output_dir, f"pca_multi3d_{args.model}_layer{args.layer}_{datasets_str}.png")
        fig3d.write_image(out3d)
        print(f"Saved 3D figure: {out3d}")


if __name__ == "__main__":
    main()
