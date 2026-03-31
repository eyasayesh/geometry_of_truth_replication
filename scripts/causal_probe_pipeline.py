#!/usr/bin/env python3
"""
Causal probe intervention pipeline (Section 6 of Marks & Tegmark, COLM 2024).

For each trained probe direction θ (normalized so that p(μ⁻ + θ) = p(μ⁺)),
intervenes on the residual stream at a given layer and measures how much the
model's TRUE/FALSE prediction changes.

Methodology:
  1. Build few-shot prompts from an eval dataset (default: sp_en_trans).
  2. Run a baseline forward pass → P(TRUE) - P(FALSE) per statement.
  3. For each probe direction θ:
       a. Run forward pass with +θ added at (layer, last token) → PD*⁻  (false→true)
       b. Run forward pass with -θ added at (layer, last token) → PD*⁺  (true→false)
  4. Compute Normalized Indirect Effects (NIEs):
       NIE false→true = (PD*⁻ - PD⁻) / (PD⁺ - PD⁻)
       NIE true→false = (PD*⁺ - PD⁺) / (PD⁻ - PD⁺)

     where PD⁺/PD⁻ are mean baseline PDs over true/false statements, and
     PD*⁺/PD*⁻ are the corresponding means under the respective intervention.

Probe direction normalization:
  θ = α · d  where d = probe.direction (unit vector)
  α = d · (μ⁺ - μ⁻)   (makes p(μ⁻ + θ) = p(μ⁺) for both LR and MM probes)

Results saved to:
  {output_dir}/{model}/layer_{layer}/{train_group}/results.json

Config format:
  {
    "eval_dataset": "sp_en_trans",
    "few_shot_examples": [
        {"statement": "The Spanish word 'fruta' means 'goat'.", "label": 0},
        {"statement": "The Spanish word 'carne' means 'meat'.", "label": 1}
    ],
    "batch_size": 25,
    "seed": 42,
    "models": {
        "llama-3.1-8b-instruct": {"layers": [8, 12, 15]}
    },
    "train_groups": [
        ["cities"],
        ["cities", "neg_cities"],
        ["larger_than"],
        ["larger_than", "smaller_than"]
    ]
  }

Example:
    python scripts/causal_probe_pipeline.py \\
        --config data/configs/probes/causal_probe_config.json \\
        --probes_dir /scratch/geometry_of_truth/probes \\
        --acts_dir   /scratch/geometry_of_truth/acts \\
        --output_dir /scratch/geometry_of_truth/causal_interventions
"""

import argparse
import gc
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations import resid_post_hook
from src.data import load_dataset
from src.models import MODEL_REGISTRY, load_model
from src.probes import LRProbe, MassMeanProbe


# ── Probe loading ─────────────────────────────────────────────────────────────

def load_probes(
    probes_dir: str,
    model_name: str,
    layer: int,
    train_key: str,
) -> dict[str, "LRProbe | MassMeanProbe | None"]:
    """
    Load lr, mm, and mm_cov probes from {probes_dir}/{model}/layer_{layer}/probes/{train_key}/.
    Returns a dict; missing probe files are set to None (with a warning).
    """
    probe_dir = os.path.join(probes_dir, model_name, f"layer_{layer}", "probes", train_key)
    probe_files = {
        "lr":     (os.path.join(probe_dir, "lr.pt"),     LRProbe),
        "mm":     (os.path.join(probe_dir, "mm.pt"),     MassMeanProbe),
        "mm_cov": (os.path.join(probe_dir, "mm_cov.pt"), MassMeanProbe),
    }
    probes = {}
    for name, (path, cls) in probe_files.items():
        if not os.path.exists(path):
            print(f"    WARNING: probe not found: {path} — skipping '{name}'")
            probes[name] = None
        else:
            probes[name] = cls.load(path)
    return probes


# ── Activation helpers ────────────────────────────────────────────────────────

def load_acts_and_labels(
    model_name: str,
    dataset_name: str,
    layer: int,
    acts_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from src.activations import load_acts
    acts = load_acts(model_name, dataset_name, layer, output_dir=acts_dir, center=True)
    labels_path = os.path.join(acts_dir, model_name, dataset_name, "labels.pt")
    if os.path.exists(labels_path):
        labels = torch.load(labels_path, weights_only=True).long()
    else:
        _, raw_labels = load_dataset(dataset_name)
        labels = torch.tensor(raw_labels, dtype=torch.long)
    n = min(len(acts), len(labels))
    return acts[:n], labels[:n]


def load_group_acts_and_labels(
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
    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)


def compute_theta(
    probe: "LRProbe | MassMeanProbe",
    acts: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute normalized probe direction θ = α · d such that p(μ⁻ + θ) = p(μ⁺).
    α = d · (μ⁺ - μ⁻)  (exact for both LR and MM probes).
    """
    d       = probe.direction.float()
    mu_pos  = acts[labels == 1].float().mean(dim=0)
    mu_neg  = acts[labels == 0].float().mean(dim=0)
    alpha   = float(d @ (mu_pos - mu_neg))
    return alpha * d


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompts(
    statements: list[str],
    few_shot_examples: list[dict],
) -> list[str]:
    """
    Build few-shot prompts: context examples followed by each test statement.

    Format per context example:
      {statement} This statement is: {TRUE|FALSE}
    Test line:
      {statement} This statement is:
    """
    prefix_lines = []
    for ex in few_shot_examples:
        label = "TRUE" if ex["label"] == 1 else "FALSE"
        prefix_lines.append(f"{ex['statement']} This statement is: {label}")
    prefix = "\n".join(prefix_lines) + "\n" if prefix_lines else ""
    return [f"{prefix}{s} This statement is:" for s in statements]


def get_true_false_ids(model) -> tuple[int, int]:
    """Return (true_id, false_id) for tokens ' TRUE' and ' FALSE'."""
    true_id  = int(model.to_tokens(" TRUE",  prepend_bos=False)[0, 0])
    false_id = int(model.to_tokens(" FALSE", prepend_bos=False)[0, 0])
    return true_id, false_id


# ── Forward pass helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_pd_batch(
    model,
    tokens: torch.Tensor,
    true_id: int,
    false_id: int,
    delta: "torch.Tensor | None",
    layer: "int | None",
) -> torch.Tensor:
    """
    Run a forward pass on tokens; if delta is given, add it to the residual
    stream at (layer, last token) before continuing downstream.

    Returns [batch] tensor of P(TRUE) - P(FALSE) at the last token position.
    """
    if delta is not None:
        def hook_fn(value, hook):
            value[:, -1, :] = value[:, -1, :] + delta
            return value
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(resid_post_hook(layer), hook_fn)],
        )
    else:
        logits = model(tokens)

    probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
    return (probs[:, true_id] - probs[:, false_id]).cpu()


def run_pd_over_dataset(
    model,
    prompts: list[str],
    true_id: int,
    false_id: int,
    delta: "torch.Tensor | None",
    layer: "int | None",
    batch_size: int,
) -> torch.Tensor:
    """Run compute_pd_batch over all prompts in mini-batches."""
    model.tokenizer.padding_side = "left"
    all_pd = []
    for i in tqdm(range(0, len(prompts), batch_size), leave=False):
        batch = prompts[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        pd = compute_pd_batch(model, tokens, true_id, false_id, delta, layer)
        all_pd.append(pd)
    return torch.cat(all_pd, dim=0)


# ── NIE computation ───────────────────────────────────────────────────────────

def compute_nie(
    pd_pos: torch.Tensor,      # baseline PD for true statements
    pd_neg: torch.Tensor,      # baseline PD for false statements
    pd_star_pos: torch.Tensor, # PD after -θ intervention on true statements
    pd_star_neg: torch.Tensor, # PD after +θ intervention on false statements
) -> dict[str, float]:
    PD_pos      = float(pd_pos.mean())
    PD_neg      = float(pd_neg.mean())
    PD_star_pos = float(pd_star_pos.mean())
    PD_star_neg = float(pd_star_neg.mean())

    denom_f2t = PD_pos - PD_neg
    denom_t2f = PD_neg - PD_pos

    nie_f2t = (PD_star_neg - PD_neg) / denom_f2t if abs(denom_f2t) > 1e-8 else float("nan")
    nie_t2f = (PD_star_pos - PD_pos) / denom_t2f if abs(denom_t2f) > 1e-8 else float("nan")

    return {
        "PD_pos":            round(PD_pos,      4),
        "PD_neg":            round(PD_neg,      4),
        "PD_star_pos":       round(PD_star_pos, 4),
        "PD_star_neg":       round(PD_star_neg, 4),
        "NIE_false_to_true": round(nie_f2t,     4),
        "NIE_true_to_false": round(nie_t2f,     4),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_model(
    model_name: str,
    layers: list[int],
    train_groups: list[tuple[str, ...]],
    eval_dataset: str,
    few_shot_examples: list[dict],
    probes_dir: str,
    acts_dir: str,
    output_dir: str,
    batch_size: int = 25,
) -> None:
    model, config = load_model(model_name)
    device = next(model.parameters()).device

    statements, labels_list = load_dataset(eval_dataset)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    pos_mask = labels_tensor == 1
    neg_mask = labels_tensor == 0

    prompts  = build_prompts(statements, few_shot_examples)
    true_id, false_id = get_true_false_ids(model)
    print(f"  TRUE token id: {true_id} ('{model.tokenizer.decode([true_id])}')")
    print(f"  FALSE token id: {false_id} ('{model.tokenizer.decode([false_id])}')")
    print(f"  Eval dataset: {eval_dataset}  |  n={len(statements)} ({pos_mask.sum()} true, {neg_mask.sum()} false)")

    # Compute baseline PD once (no hooks needed)
    print("\n  Computing baseline PD…")
    pd_baseline = run_pd_over_dataset(
        model, prompts, true_id, false_id, delta=None, layer=None, batch_size=batch_size
    )

    for layer in layers:
        print(f"\n  Layer {layer}")
        layer_results = {}

        for group in train_groups:
            train_key = "+".join(group)
            print(f"    Train group: {train_key}")

            probes = load_probes(probes_dir, model_name, layer, train_key)
            if all(p is None for p in probes.values()):
                print(f"    WARNING: no probes found — skipping '{train_key}'")
                continue

            # Load training activations for probe direction normalization
            try:
                train_acts, train_labels = load_group_acts_and_labels(
                    model_name, group, layer, acts_dir
                )
            except FileNotFoundError as e:
                print(f"    WARNING: {e} — skipping '{train_key}'")
                continue

            probe_results = {}
            for probe_name, probe in probes.items():
                if probe is None:
                    continue

                theta = compute_theta(probe, train_acts, train_labels).to(device)
                theta_norm = float(theta.norm())

                # +θ intervention (false→true): add θ to all statements
                pd_plus = run_pd_over_dataset(
                    model, prompts, true_id, false_id, delta=theta, layer=layer,
                    batch_size=batch_size
                )
                # -θ intervention (true→false): subtract θ from all statements
                pd_minus = run_pd_over_dataset(
                    model, prompts, true_id, false_id, delta=-theta, layer=layer,
                    batch_size=batch_size
                )

                # Split by ground-truth label
                nie_metrics = compute_nie(
                    pd_pos      = pd_baseline[pos_mask],   # baseline true statements
                    pd_neg      = pd_baseline[neg_mask],   # baseline false statements
                    pd_star_pos = pd_minus[pos_mask],      # true statements, -θ applied
                    pd_star_neg = pd_plus[neg_mask],       # false statements, +θ applied
                )
                nie_metrics["theta_norm"] = round(theta_norm, 4)

                probe_results[probe_name] = nie_metrics
                print(
                    f"      [{probe_name}]  NIE f→t: {nie_metrics['NIE_false_to_true']:+.3f}  "
                    f"NIE t→f: {nie_metrics['NIE_true_to_false']:+.3f}"
                )

            layer_results[train_key] = {
                "layer":           layer,
                "model":           model_name,
                "eval_dataset":    eval_dataset,
                "n_true":          int(pos_mask.sum()),
                "n_false":         int(neg_mask.sum()),
                "PD_pos_baseline": round(float(pd_baseline[pos_mask].mean()), 4),
                "PD_neg_baseline": round(float(pd_baseline[neg_mask].mean()), 4),
                "probes":          probe_results,
            }

        # Save per-layer results
        results_path = os.path.join(output_dir, model_name, f"layer_{layer}", "results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(layer_results, f, indent=4)
        print(f"\n    Saved: {results_path}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("  GPU memory released.")


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Causal probe intervention pipeline")
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument(
        "--probes_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/probes",
        help="Root dir for probe .pt files",
    )
    parser.add_argument(
        "--acts_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts",
        help="Root dir for saved activations",
    )
    parser.add_argument(
        "--output_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/causal_interventions",
        help="Root dir for output results",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    eval_dataset      = cfg.get("eval_dataset", "sp_en_trans")
    few_shot_examples = cfg.get("few_shot_examples", [
        {"statement": "The Spanish word 'fruta' means 'goat'.", "label": 0},
        {"statement": "The Spanish word 'carne' means 'meat'.", "label": 1},
    ])
    batch_size        = cfg.get("batch_size", 25)
    train_groups      = [tuple(g) for g in cfg.get("train_groups", [])]

    for model_name, model_cfg in cfg["models"].items():
        if model_name not in MODEL_REGISTRY:
            print(f"WARNING: unknown model '{model_name}' — skipping.")
            continue
        layers = model_cfg["layers"]
        print(f"\n{'=' * 70}")
        print(f"Model: {MODEL_REGISTRY[model_name].display_name}  |  layers: {layers}")
        print(f"{'=' * 70}")

        run_model(
            model_name        = model_name,
            layers            = layers,
            train_groups      = train_groups,
            eval_dataset      = eval_dataset,
            few_shot_examples = few_shot_examples,
            probes_dir        = args.probes_dir,
            acts_dir          = args.acts_dir,
            output_dir        = args.output_dir,
            batch_size        = batch_size,
        )


if __name__ == "__main__":
    main()
