#!/usr/bin/env python3
"""
End-to-end PCA pipeline: for each model, extract activations for all datasets
at specified layers (skipping any that already exist), unload the model, then
run PCA and save figures and PCA results.

This keeps peak GPU memory to one model at a time. PCA runs on CPU after the
model is unloaded.

Activations are written to  {acts_dir}/{model}/{dataset}/layer_{layer}_*.pt
PCA results are written to  {pca_dir}/{model}/{dataset}/pca_layer{layer}.pt
Figures are written to      {figures_dir}/{model}/{dataset}/pca_layer{layer}.png

Extraction is skipped for any (model, dataset, layer) where activations already
exist on disk, so the pipeline is safe to rerun after partial failures.

Example:
    python scripts/pca_pipeline.py --config data/pca_config.json

    python scripts/pca_pipeline.py \\
        --config data/pca_config.json \\
        --acts_dir /scratch/acts \\
        --pca_dir  /scratch/pca \\
        --figures_dir figures/pca
"""

import argparse
import gc
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations import extract_acts, load_acts, save_acts
from src.data import load_dataset, ALL_DATASETS
from src.models import MODEL_REGISTRY, load_model
from src.pca import run_pca, plot_pca


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> tuple[dict[str, dict], list[str], int | None, int]:
    """
    Returns:
        models:   dict mapping model_name -> {"layers": [...], "batch_size": int}
        datasets: list of dataset names
        max_rows: row cap per dataset (None = no limit)
        seed:     random seed for sampling
    """
    with open(path) as f:
        cfg = json.load(f)

    raw = cfg["models"]
    unknown = [m for m in raw if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s) in config: {unknown}")

    models = {}
    for name, entry in raw.items():
        if isinstance(entry, list):
            # backwards-compatible: plain list of layers
            models[name] = {"layers": entry, "batch_size": 25}
        else:
            models[name] = {
                "layers": entry["layers"],
                "batch_size": entry.get("batch_size", 25),
            }

    datasets = cfg.get("datasets", ALL_DATASETS)
    unknown_ds = [d for d in datasets if d not in ALL_DATASETS]
    if unknown_ds:
        raise ValueError(f"Unknown dataset(s) in config: {unknown_ds}")

    max_rows = cfg.get("max_rows", None)
    seed = cfg.get("seed", 42)

    return models, datasets, max_rows, seed


# ── Activation extraction ─────────────────────────────────────────────────────

def acts_exist(model_name: str, dataset_name: str, layer: int, acts_dir: str) -> bool:
    """Return True if at least one activation file exists for this combination."""
    import glob as _glob
    pattern = os.path.join(acts_dir, model_name, dataset_name, f"layer_{layer}_*.pt")
    return len(_glob.glob(pattern)) > 0


def sample_dataset(
    statements: list[str],
    labels: list[int],
    max_rows: int | None,
    seed: int,
) -> tuple[list[str], list[int]]:
    """Shuffle and subsample to max_rows if the dataset exceeds the limit."""
    if max_rows is None or len(statements) <= max_rows:
        return statements, labels
    rng = random.Random(seed)
    indices = list(range(len(statements)))
    rng.shuffle(indices)
    indices = indices[:max_rows]
    return [statements[i] for i in indices], [labels[i] for i in indices]


def extract_for_model(
    model_name: str,
    datasets: list[str],
    layers: list[int],
    acts_dir: str,
    batch_size: int = 25,
    max_rows: int | None = None,
    seed: int = 42,
) -> None:
    """
    Load model, extract activations for all (dataset, layer) pairs that don't
    already exist, then unload the model and free GPU memory.
    """
    needed = [
        (ds, layer)
        for ds in datasets
        for layer in layers
        if not acts_exist(model_name, ds, layer, acts_dir)
    ]

    if not needed:
        print("  All activations already extracted — skipping model load.")
        return

    needed_layers = sorted({layer for _, layer in needed})
    needed_set = {ds for ds, _ in needed}
    needed_datasets = [ds for ds in datasets if ds in needed_set]
    print(f"  Loading model to extract {len(needed)} missing (dataset, layer) pairs...")

    model, config = load_model(model_name)

    for dataset_name in needed_datasets:
        layers_for_ds = [l for l in needed_layers if not acts_exist(model_name, dataset_name, l, acts_dir)]
        if not layers_for_ds:
            continue

        try:
            statements, labels = load_dataset(dataset_name)
        except FileNotFoundError as e:
            print(f"    WARNING: {e} — skipping.")
            continue

        statements, labels = sample_dataset(statements, labels, max_rows, seed)
        n = len(statements)
        print(f"  Dataset: {dataset_name} ({n} rows) — extracting layers {layers_for_ds}")

        acts = extract_acts(model=model, statements=statements, layers=layers_for_ds, batch_size=batch_size)
        save_acts(acts, model_name, dataset_name, acts_dir, batch_size=batch_size)

        # Save labels so PCA uses the correct (possibly subsampled) subset
        labels_path = os.path.join(acts_dir, model_name, dataset_name, "labels.pt")
        torch.save(torch.tensor(labels, dtype=torch.long), labels_path)

        print(f"    Saved to {acts_dir}/{model_name}/{dataset_name}/")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("  GPU memory released.")


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca_for_model(
    model_name: str,
    datasets: list[str],
    layers: list[int],
    acts_dir: str,
    pca_dir: str,
    figures_dir: str,
    n_pcs: int = 10,
) -> None:
    config = MODEL_REGISTRY[model_name]

    for dataset_name in datasets:
        for layer in layers:
            try:
                acts = load_acts(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    layer=layer,
                    output_dir=acts_dir,
                    center=False,
                )
            except FileNotFoundError:
                print(f"    WARNING: no activations for {dataset_name} layer {layer} — skipping.")
                continue

            labels_path = os.path.join(acts_dir, model_name, dataset_name, "labels.pt")
            if os.path.exists(labels_path):
                labels = torch.load(labels_path, weights_only=True).tolist()
            else:
                _, labels = load_dataset(dataset_name)

            k = min(n_pcs, acts.shape[1], acts.shape[0] - 1)
            result = run_pca(acts, k=k)

            ev = result.explained_var_ratio
            print(f"    {dataset_name} layer {layer}: PC1={ev[0]:.1%}, PC2={ev[1]:.1%}")

            # Save PCA result
            pca_out = os.path.join(pca_dir, model_name, dataset_name)
            os.makedirs(pca_out, exist_ok=True)
            torch.save({
                "components": result.components,
                "projections": result.projections,
                "explained_var_ratio": result.explained_var_ratio,
                "model": model_name,
                "dataset": dataset_name,
                "layer": layer,
            }, os.path.join(pca_out, f"pca_layer{layer}.pt"))

            # Save figure
            fig_out = os.path.join(figures_dir, model_name, dataset_name)
            os.makedirs(fig_out, exist_ok=True)
            title = f"{config.display_name} — {dataset_name} — layer {layer}<br>"
            title += f"<sup>PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%} explained variance</sup>"
            fig = plot_pca(result, labels, title=title)
            fig.write_image(os.path.join(fig_out, f"pca_layer{layer}.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PCA pipeline across models and datasets")
    parser.add_argument(
        "--config",
        default="data/pca_config.json",
        help="JSON config file with models (and per-model layers) and datasets (default: data/pca_config.json)",
    )
    parser.add_argument(
        "--acts_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts",
        help="Root directory for activations",
    )
    parser.add_argument(
        "--pca_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/pca",
        help="Root directory for PCA result .pt files",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures/pca",
        help="Root directory for PNG figures (default: figures/pca)",
    )
    parser.add_argument(
        "--n_pcs",
        type=int,
        default=10,
        help="Number of PCs to compute and report (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models, datasets, max_rows, seed = load_config(args.config)

    print(f"Models:   {list(models.keys())}")
    print(f"Datasets: {datasets}")
    print(f"max_rows: {max_rows}  |  seed: {seed}")
    print()

    for model_name, model_cfg in models.items():
        layers = model_cfg["layers"]
        batch_size = model_cfg["batch_size"]
        print(f"{'=' * 60}")
        print(f"Model: {MODEL_REGISTRY[model_name].display_name}  |  layers: {layers}  |  batch_size: {batch_size}")
        print(f"{'=' * 60}")

        # Step 1: extract activations (GPU)
        try:
            extract_for_model(model_name, datasets, layers, args.acts_dir, batch_size=batch_size, max_rows=max_rows, seed=seed)
        except Exception as e:
            print(f"  ERROR during extraction: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            print()
            continue

        # Step 2: PCA + figures (CPU)
        print(f"\n  Running PCA and generating figures...")
        try:
            run_pca_for_model(
                model_name=model_name,
                datasets=datasets,
                layers=model_cfg["layers"],
                acts_dir=args.acts_dir,
                pca_dir=args.pca_dir,
                figures_dir=args.figures_dir,
                n_pcs=args.n_pcs,
            )
        except Exception as e:
            print(f"  ERROR during PCA: {type(e).__name__}: {e}")

        print()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
