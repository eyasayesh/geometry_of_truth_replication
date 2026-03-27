#!/usr/bin/env python3
"""
Run activation patching experiments from prompt pairs defined in a JSON file.

Input JSON format:
    [
        {
            "baseline_prompt":  "The Spanish word 'uno' means 'one'. This statement is:",
            "corrupted_prompt": "The Spanish word 'con' means 'one'. This statement is:",
            "baseline_answer":  " TRUE",
            "corrupted_answer": " FALSE"
        },
        ...
    ]

Results are appended to an output JSON file with one entry per prompt pair:
    [
        {
            "model":                "llama-3.2-1b",
            "baseline_prompt":      "...",
            "corrupted_prompt":     "...",
            "baseline_answer":      " TRUE",
            "corrupted_answer":     " FALSE",
            "n_toks":               1,
            "baseline_logit_diff":  2.1,
            "corrupted_logit_diff": -2.3,
            "logit_diffs":          [[...], ...]   # [n_toks][n_layers]
        },
        ...
    ]

Example:
    python scripts/patching_experiment.py \\
        --model llama-3.2-1b \\
        --prompts_file data/patching_prompts/sp_en_trans.json \\
        --output_file experimental_outputs/patching_sp_en_trans.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import load_model, MODEL_REGISTRY
from src.patching import run_patching_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run activation patching experiments")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use",
    )
    parser.add_argument(
        "--prompts_file",
        required=True,
        help="Path to JSON file containing list of {true_prompt, false_prompt} pairs",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Path to output JSON file (default: experimental_outputs/patching_{dataset}.json)",
    )
    return parser.parse_args()


def load_prompt_pairs(path: str) -> list[dict]:
    with open(path) as f:
        pairs = json.load(f)
    if not isinstance(pairs, list):
        pairs = [pairs]
    required = {"baseline_prompt", "corrupted_prompt", "baseline_answer", "corrupted_answer"}
    for pair in pairs:
        missing = required - pair.keys()
        if missing:
            raise ValueError(
                f"Each entry must have {required}. Missing: {missing}"
            )
    return pairs


def warn_if_exists(path: str) -> None:
    if os.path.exists(path):
        print(f"WARNING: '{path}' already exists and will be overwritten.")


def save_results(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


def main():
    args = parse_args()

    prompt_pairs = load_prompt_pairs(args.prompts_file)
    print(f"Loaded {len(prompt_pairs)} prompt pair(s) from {args.prompts_file}")

    dataset_name = os.path.splitext(os.path.basename(args.prompts_file))[0]
    output_file = args.output_file or f"experimental_outputs/patching_{dataset_name}.json"

    model, config = load_model(args.model)
    print(f"Loaded {config.display_name} ({config.n_layers} layers)")

    warn_if_exists(output_file)
    results = []

    for i, pair in enumerate(prompt_pairs):
        print(f"\n--- Pair {i + 1}/{len(prompt_pairs)} ---")
        print(f"  BASELINE:  {pair['baseline_prompt'][:80]}...")
        print(f"  CORRUPTED: {pair['corrupted_prompt'][:80]}...")

        result = run_patching_experiment(
            model=model,
            baseline_prompt=pair["baseline_prompt"],
            corrupted_prompt=pair["corrupted_prompt"],
            baseline_answer=pair["baseline_answer"],
            corrupted_answer=pair["corrupted_answer"],
        )
        result["model"] = config.name
        results.append(result)

        # Save incrementally so progress is not lost on failure
        save_results(results, output_file)
        print(f"  n_toks={result['n_toks']}, corrupted logit diff={result['corrupted_logit_diff']:.3f}")

    print(f"\nDone. Results saved to {output_file}")


if __name__ == "__main__":
    main()
