#!/usr/bin/env python3
"""
Generate and cache residual-stream activations for truth probing experiments.

Example:
    python scripts/generate_acts.py \
        --model llama-3.2-1b \
        --datasets cities neg_cities sp_en_trans \
        --layers 0 4 8 12 15
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import load_model, MODEL_REGISTRY
from src.activations import extract_acts, save_acts, BATCH_SIZE
from src.data import load_dataset, ALL_DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description="Generate activations for truth probing")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use (must be in MODEL_REGISTRY)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help="Dataset names (default: all datasets)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layer indices to extract (default: all layers)",
    )
    parser.add_argument(
        "--output_dir",
        default="acts",
        help="Root directory for saving activations (default: acts/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for inference (default: {BATCH_SIZE})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model, config = load_model(args.model)
    print(f"Loaded {config.display_name} ({config.n_layers} layers, hidden={config.hidden_size})")

    layers = args.layers if args.layers is not None else list(range(config.n_layers))
    print(f"Extracting layers: {layers}")

    for dataset_name in args.datasets:
        print(f"\n--- Dataset: {dataset_name} ---")
        try:
            statements, _ = load_dataset(dataset_name)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        acts = extract_acts(
            model=model,
            statements=statements,
            layers=layers,
            batch_size=args.batch_size,
        )
        save_acts(acts, args.model, dataset_name, args.output_dir, args.batch_size)
        print(f"  Saved to {args.output_dir}/{args.model}/{dataset_name}/")


if __name__ == "__main__":
    main()
