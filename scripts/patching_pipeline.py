#!/usr/bin/env python3
"""
End-to-end patching pipeline: for each model, run all patching prompt files,
save results, generate figures, then unload the model before moving on.

This keeps peak GPU memory to one model at a time.

Results are written to  {output_dir}/{model}/patching_{dataset}.json
Figures are written to  {figures_dir}/{model}/{dataset}_pair{N}.png

If {output_dir}/{model}/ already exists the model is skipped entirely
(with a warning), so the pipeline is safe to rerun after partial failures.

Example:
    python scripts/patching_pipeline.py

    python scripts/patching_pipeline.py \\
        --models_file data/patching_models.json \\
        --prompts_dir data/patching_prompts \\
        --output_dir  experimental_outputs \\
        --figures_dir figures/patching
"""

import argparse
import gc
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MODEL_REGISTRY, load_model
from src.patching import run_patching_experiment
from src.visualization import plot_all_patching_results


def parse_args():
    parser = argparse.ArgumentParser(description="Patching pipeline across models and datasets")
    parser.add_argument(
        "--models_file",
        default="data/patching_models.json",
        help="JSON file containing list of model names to run (default: data/patching_models.json)",
    )
    parser.add_argument(
        "--prompts_dir",
        default="data/patching_prompts",
        help="Directory containing patching prompt JSON files (default: data/patching_prompts)",
    )
    parser.add_argument(
        "--output_dir",
        default="experimental_outputs",
        help="Root directory for result JSON files; one sub-dir per model (default: experimental_outputs)",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures/patching",
        help="Root directory for PNG figures; one sub-dir per model (default: figures/patching)",
    )
    return parser.parse_args()


def load_models_list(path: str) -> list[str]:
    with open(path) as f:
        models = json.load(f)
    if not isinstance(models, list):
        raise ValueError(f"{path} must contain a JSON array of model name strings.")
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s) in {path}: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")
    return models


def load_prompt_pairs(path: str) -> list[dict]:
    with open(path) as f:
        pairs = json.load(f)
    if not isinstance(pairs, list):
        pairs = [pairs]
    return pairs


def save_results(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


def run_model(model_name: str, prompt_files: list[str], model_output_dir: str) -> list[str]:
    """
    Load model, run all prompt files, return list of result JSON paths written.
    Unloads the model from GPU before returning.
    """
    model, config = load_model(model_name)
    result_paths = []

    for prompts_path in prompt_files:
        dataset = os.path.splitext(os.path.basename(prompts_path))[0]
        output_path = os.path.join(model_output_dir, f"patching_{dataset}.json")

        pairs = load_prompt_pairs(prompts_path)
        print(f"  Running {dataset} ({len(pairs)} pair(s))…")
        results = []

        for i, pair in enumerate(pairs):
            print(f"    Pair {i + 1}/{len(pairs)}")
            try:
                result = run_patching_experiment(
                    model=model,
                    baseline_prompt=pair["baseline_prompt"],
                    corrupted_prompt=pair["corrupted_prompt"],
                    baseline_answer=pair["baseline_answer"],
                    corrupted_answer=pair["corrupted_answer"],
                )
            except ValueError as e:
                print(f"    WARNING: skipping pair {i + 1} — {e}")
                continue
            result["model"] = config.name
            results.append(result)
            save_results(results, output_path)  # incremental save

        print(f"  Saved {output_path}")
        result_paths.append(output_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("  GPU memory released.")

    return result_paths


def generate_figures(result_paths: list[str], model_name: str, model_figures_dir: str) -> None:
    """Load tokenizer (CPU only) and render one PNG per result pair."""
    from transformers import AutoTokenizer

    hf_id = MODEL_REGISTRY[model_name].hf_id
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    tokenizer.padding_side = "left"

    os.makedirs(model_figures_dir, exist_ok=True)

    for result_path in result_paths:
        with open(result_path) as f:
            results = json.load(f)
        if not isinstance(results, list):
            results = [results]

        dataset = os.path.splitext(os.path.basename(result_path))[0]
        if dataset.startswith("patching_"):
            dataset = dataset[len("patching_"):]

        figures = plot_all_patching_results(results, tokenizer=tokenizer)

        for i, fig in enumerate(figures):
            png_path = os.path.join(model_figures_dir, f"{dataset}_pair{i + 1}.png")
            fig.write_image(png_path)
            print(f"  Saved {png_path}")


def main():
    args = parse_args()

    models = load_models_list(args.models_file)

    prompt_files = sorted(glob.glob(os.path.join(args.prompts_dir, "*.json")))
    if not prompt_files:
        print(f"No JSON files found in {args.prompts_dir}")
        sys.exit(1)

    print(f"Models:  {models}")
    print(f"Prompts: {[os.path.basename(p) for p in prompt_files]}")
    print()

    for model_name in models:
        print(f"{'=' * 60}")
        print(f"Model: {MODEL_REGISTRY[model_name].display_name}")
        print(f"{'=' * 60}")

        model_output_dir = os.path.join(args.output_dir, model_name)
        model_figures_dir = os.path.join(args.figures_dir, model_name)

        if os.path.isdir(model_output_dir):
            print(f"  WARNING: output directory '{model_output_dir}' already exists — skipping model.")
            print()
            continue

        try:
            result_paths = run_model(
                model_name=model_name,
                prompt_files=prompt_files,
                model_output_dir=model_output_dir,
            )
        except Exception as e:
            print(f"\n  WARNING: failed to process {MODEL_REGISTRY[model_name].display_name} — skipping.")
            print(f"  Reason: {type(e).__name__}: {e}")
            # Remove the output directory only if it is empty so the model is
            # not mistakenly treated as complete on the next run.
            # If it contains partial results, leave it and warn the user.
            if os.path.isdir(model_output_dir):
                if not os.listdir(model_output_dir):
                    os.rmdir(model_output_dir)
                else:
                    print(f"  Partial results left in '{model_output_dir}' — remove it manually to retry.")
            torch.cuda.empty_cache()
            gc.collect()
            print()
            continue

        print(f"\n  Generating figures…")
        try:
            generate_figures(
                result_paths=result_paths,
                model_name=model_name,
                model_figures_dir=model_figures_dir,
            )
        except Exception as e:
            print(f"\n  WARNING: figure generation failed for {MODEL_REGISTRY[model_name].display_name}.")
            print(f"  Reason: {type(e).__name__}: {e}")
            print(f"  Results were saved; re-run visualize_patching.py separately to retry figures.")
        print()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
