#!/usr/bin/env python3
"""
Visualize activation patching results as logit-difference heatmaps.

Reads one or more patching result JSON files produced by
`scripts/patching_experiment.py` and writes one PNG heatmap per prompt pair.

Example:
    python scripts/visualize_patching.py \\
        --results_file experimental_outputs/patching_cities.json \\
        --model llama-3.1-8b \\
        --output_dir figures/patching
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import plot_all_patching_results


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize activation patching heatmaps")
    parser.add_argument(
        "--results_file",
        required=True,
        nargs="+",
        help="Path(s) to patching result JSON file(s)",
    )
    parser.add_argument(
        "--output_dir",
        default="figures/patching",
        help="Directory for output figures (default: figures/patching)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model short name (e.g. llama-3.1-8b) used to load the tokenizer for "
            "human-readable x-axis labels. If omitted, position indices are used."
        ),
    )
    parser.add_argument(
        "--max_plots",
        type=int,
        default=None,
        help="Limit to the first N results per file",
    )
    return parser.parse_args()


def load_tokenizer(model_name: str):
    """Load only the tokenizer (no GPU needed) to decode token labels."""
    from src.models import MODEL_REGISTRY
    from transformers import AutoTokenizer

    if model_name not in MODEL_REGISTRY:
        print(
            f"WARNING: model '{model_name}' not in MODEL_REGISTRY; "
            "falling back to position indices for x-axis labels."
        )
        return None

    hf_id = MODEL_REGISTRY[model_name].hf_id
    hf_token = os.environ.get("HF_TOKEN")
    print(f"Loading tokenizer for {model_name} ({hf_id})…")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    tokenizer.padding_side = "left"
    return tokenizer


def main():
    args = parse_args()

    tokenizer = None
    if args.model:
        tokenizer = load_tokenizer(args.model)

    os.makedirs(args.output_dir, exist_ok=True)

    for results_path in args.results_file:
        print(f"\nProcessing {results_path}")
        with open(results_path) as f:
            results = json.load(f)
        if not isinstance(results, list):
            results = [results]

        dataset_name = os.path.splitext(os.path.basename(results_path))[0]
        if dataset_name.startswith("patching_"):
            dataset_name = dataset_name[len("patching_"):]

        figures = plot_all_patching_results(
            results,
            tokenizer=tokenizer,
            max_plots=args.max_plots,
        )

        model_tag = results[0].get("model", "unknown").replace("/", "-")

        for i, fig in enumerate(figures):
            stem = f"{model_tag}_{dataset_name}_pair{i + 1}"
            png_path = os.path.join(args.output_dir, f"{stem}.png")
            fig.write_image(png_path)
            print(f"  Saved {png_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
