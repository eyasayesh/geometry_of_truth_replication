#!/usr/bin/env python3
"""
PCA pipeline using the original geometry-of-truth code for activation extraction
and PCA, as a baseline for comparing against our TransformerLens implementation.

Activation extraction uses nnsight (as in the original generate_acts.py).
PCA uses get_pcs from the original utils.py.
Figures are saved using our plot_pca wrapper.

Activations written to: {acts_dir}/{model_name}/{dataset}/layer_{layer}_{idx}.pt
PCA results written to: {pca_dir}/{model_name}/{dataset}/pca_layer{layer}.pt
Figures written to:     {figures_dir}/{model_name}/{dataset}/pca_layer{layer}.png

Example:
    python scripts/original_pca_pipeline.py \\
        --model meta-llama/Llama-3.2-1B \\
        --datasets cities neg_cities \\
        --layers 8 12 \\
        --model_name llama-3.2-1b

    python scripts/original_pca_pipeline.py \\
        --model EleutherAI/pythia-410m \\
        --arch pythia \\
        --datasets cities sp_en_trans \\
        --layers 10 16
"""

import argparse
import gc
import json
import os
import sys

import torch
from glob import glob
from tqdm import tqdm

# Allow imports from repo root and geometry-of-truth
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_DIR = os.path.join(REPO_ROOT, "geometry-of-truth")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ORIG_DIR)

from nnsight import LanguageModel
from utils import get_pcs  # original paper's PCA implementation

from src.data import load_dataset, ALL_DATASETS
from src.pca import plot_pca, PCAResult

BATCH_SIZE = 64

# ── Architecture helpers ───────────────────────────────────────────────────────

# Maps arch shorthand -> function that retrieves the i-th transformer layer
# from an nnsight-traced model. The callable receives (model, i) and returns
# the layer module used inside a trace() context.
ARCH_LAYER_FN = {
    # LLaMA / Mistral / Gemma: model.model.layers[i]
    "llama":  lambda m, i: m.model.layers[i],
    "gemma":  lambda m, i: m.model.layers[i],
    # GPT-2: model.transformer.h[i]
    "gpt2":   lambda m, i: m.transformer.h[i],
    # OPT: model.model.decoder.layers[i]
    "opt":    lambda m, i: m.model.decoder.layers[i],
    # Pythia / GPT-NeoX: model.gpt_neox.layers[i]
    "pythia": lambda m, i: m.gpt_neox.layers[i],
}

ARCH_ALIASES = {
    "llama":   "llama",
    "mistral": "llama",
    "gemma":   "gemma",
    "gpt2":    "gpt2",
    "opt":     "opt",
    "pythia":  "pythia",
    "neox":    "pythia",
    "gptneox": "pythia",
}


def resolve_arch(arch_str: str) -> str:
    key = arch_str.lower().replace("-", "").replace("_", "")
    if key not in ARCH_ALIASES:
        raise ValueError(
            f"Unknown architecture '{arch_str}'. "
            f"Choose from: {list(ARCH_ALIASES.keys())}"
        )
    return ARCH_ALIASES[key]


def infer_arch(hf_id: str) -> str:
    """Best-effort arch inference from the HF model ID."""
    lower = hf_id.lower()
    if "llama" in lower or "mistral" in lower:
        return "llama"
    if "gemma" in lower:
        return "gemma"
    if "gpt2" in lower or "gpt-2" in lower:
        return "gpt2"
    if "opt" in lower:
        return "opt"
    if "pythia" in lower or "neox" in lower:
        return "pythia"
    raise ValueError(
        f"Cannot infer architecture from '{hf_id}'. "
        f"Please pass --arch explicitly. Options: {list(ARCH_ALIASES.keys())}"
    )


# ── Activation extraction (original nnsight approach) ─────────────────────────

def get_acts_nnsight(
    statements: list[str],
    model: LanguageModel,
    layers: list[int],
    arch: str,
) -> dict[int, torch.Tensor]:
    """
    Extract last-token activations via nnsight, mirroring generate_acts.get_acts().

    Returns dict: layer -> [n_statements, hidden_size] tensor.

    NOTE: the original code uses `layer.output[0][:, -1, :]` which grabs position -1
    regardless of padding. For variable-length batches this may capture padding tokens
    (right-padding is the HuggingFace default). This matches the original's behaviour.

    NOTE: in newer nnsight versions, layer.output is the hidden states tensor directly
    rather than a tuple, so we use layer.output[:, -1, :] instead of output[0][:, -1, :].
    NOTE: in newer nnsight versions, .save() returns the tensor directly after the trace
    context exits rather than a proxy with a .value attribute.
    """
    layer_fn = ARCH_LAYER_FN[arch]
    saved = {}

    with model.trace(statements):
        for layer in layers:
            saved[layer] = layer_fn(model, layer).output[:, -1, :].save()

    return {layer: saved[layer].detach().cpu() for layer in layers}


def acts_exist(model_name: str, dataset: str, layer: int, acts_dir: str) -> bool:
    pattern = os.path.join(acts_dir, model_name, dataset, f"layer_{layer}_*.pt")
    return len(glob(pattern)) > 0


def extract_for_model(
    hf_id: str,
    model_name: str,
    arch: str,
    datasets: list[str],
    layers: list[int],
    acts_dir: str,
    noperiod: bool = False,
) -> None:
    needed = [
        (ds, layer)
        for ds in datasets
        for layer in layers
        if not acts_exist(model_name, ds, layer, acts_dir)
    ]

    if not needed:
        print("  All activations already exist — skipping model load.")
        return

    print(f"  Loading {hf_id} with nnsight (local)...")
    model = LanguageModel(hf_id, dtype=torch.bfloat16, device_map="auto")

    needed_layers = sorted({layer for _, layer in needed})
    needed_datasets = list(dict.fromkeys(ds for ds, _ in needed))

    for dataset_name in needed_datasets:
        layers_for_ds = [
            l for l in needed_layers
            if not acts_exist(model_name, dataset_name, l, acts_dir)
        ]
        if not layers_for_ds:
            continue

        try:
            statements, labels = load_dataset(dataset_name)
        except FileNotFoundError as e:
            print(f"    WARNING: {e} — skipping.")
            continue

        if noperiod:
            statements = [s.rstrip(".") for s in statements]

        save_dir = os.path.join(acts_dir, model_name, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"  Dataset: {dataset_name} ({len(statements)} rows) — layers {layers_for_ds}")

        for idx in tqdm(range(0, len(statements), BATCH_SIZE), desc=f"  {dataset_name}"):
            batch = statements[idx : idx + BATCH_SIZE]
            acts = get_acts_nnsight(batch, model, layers_for_ds, arch)
            for layer, act in acts.items():
                torch.save(act, os.path.join(save_dir, f"layer_{layer}_{idx}.pt"))

        # Save labels alongside activations
        torch.save(
            torch.tensor(labels, dtype=torch.long),
            os.path.join(save_dir, "labels.pt"),
        )
        print(f"    Saved to {save_dir}/")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("  GPU memory released.")


# ── Load activations ──────────────────────────────────────────────────────────

def load_acts(
    model_name: str,
    dataset: str,
    layer: int,
    acts_dir: str,
) -> torch.Tensor:
    """Load and concatenate all batched activation chunks for a layer."""
    directory = os.path.join(acts_dir, model_name, dataset)
    files = sorted(
        glob(os.path.join(directory, f"layer_{layer}_*.pt")),
        key=lambda f: int(os.path.basename(f).split("_")[-1].replace(".pt", "")),
    )
    if not files:
        raise FileNotFoundError(
            f"No activations at '{directory}' for layer {layer}."
        )
    return torch.cat(
        [torch.load(f, weights_only=True) for f in files], dim=0
    ).detach().float()


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca_original(acts: torch.Tensor, k: int = 2) -> PCAResult:
    """
    Run PCA using the original paper's get_pcs() from utils.py.
    get_pcs centers the data internally.
    Returns a PCAResult compatible with our plot_pca().
    """
    components = get_pcs(acts, k=k)  # [hidden_size, k], data is centered inside

    acts_centered = acts - acts.mean(dim=0)
    projections = torch.mm(acts_centered, components)   # [n, k]

    # Compute explained variance ratios from the components
    cov = torch.mm(acts_centered.T, acts_centered) / (acts_centered.shape[0] - 1)
    eigenvalues, _ = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.flip(0)
    total_var = eigenvalues.sum().item()
    explained_var_ratio = [
        (torch.mm(components[:, i:i+1].T, torch.mm(cov, components[:, i:i+1])).item() / total_var)
        for i in range(k)
    ]

    return PCAResult(
        projections=projections.detach(),
        components=components,
        explained_var_ratio=explained_var_ratio,
    )


def run_pca_for_model(
    model_name: str,
    display_name: str,
    datasets: list[str],
    layers: list[int],
    acts_dir: str,
    pca_dir: str,
    figures_dir: str,
    n_pcs: int = 10,
) -> None:
    for dataset_name in datasets:
        for layer in layers:
            try:
                acts = load_acts(model_name, dataset_name, layer, acts_dir)
            except FileNotFoundError:
                print(f"    WARNING: no activations for {dataset_name} layer {layer} — skipping.")
                continue

            labels_path = os.path.join(acts_dir, model_name, dataset_name, "labels.pt")
            if os.path.exists(labels_path):
                labels = torch.load(labels_path, weights_only=True).tolist()
            else:
                _, labels = load_dataset(dataset_name)

            k = min(n_pcs, acts.shape[1], acts.shape[0] - 1)
            result = run_pca_original(acts, k=k)

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
            title = f"{display_name} (original) — {dataset_name} — layer {layer}<br>"
            title += f"<sup>PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%} explained variance</sup>"
            fig = plot_pca(result, labels, title=title)
            fig.write_image(os.path.join(fig_out, f"pca_layer{layer}.png"))


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> tuple[dict, list[str], bool]:
    """
    Load a JSON config file.

    Config format:
    {
        "models": {
            "llama-2-13b-hf": {
                "hf_id": "meta-llama/Llama-2-13b-hf",
                "arch": "llama",          // optional, inferred from hf_id if omitted
                "layers": [8, 9, ..., 20],
                "batch_size": 64          // optional, default 64
            }
        },
        "datasets": [...],                // optional, default all
        "noperiod": false,                // optional, default false
        "seed": 42                        // optional, default 42
    }

    Returns:
        models:   dict of model_name -> {hf_id, arch, layers, batch_size}
        datasets: list of dataset names
        noperiod: bool
    """
    with open(path) as f:
        cfg = json.load(f)

    models = {}
    for name, entry in cfg["models"].items():
        if "hf_id" not in entry:
            raise ValueError(f"Model '{name}' in config is missing required 'hf_id' field.")
        arch = entry.get("arch", None)
        if arch:
            arch = resolve_arch(arch)
        else:
            arch = infer_arch(entry["hf_id"])
        models[name] = {
            "hf_id":      entry["hf_id"],
            "arch":       arch,
            "layers":     entry["layers"],
            "batch_size": entry.get("batch_size", BATCH_SIZE),
        }

    datasets = cfg.get("datasets", ALL_DATASETS)
    unknown_ds = [d for d in datasets if d not in ALL_DATASETS]
    if unknown_ds:
        raise ValueError(f"Unknown dataset(s) in config: {unknown_ds}")

    noperiod = cfg.get("noperiod", False)

    return models, datasets, noperiod


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA pipeline using original geometry-of-truth code (nnsight)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "JSON config file specifying multiple models/layers/batch sizes. "
            "When provided, --model and --layers are ignored."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID, e.g. meta-llama/Llama-3.2-1B (single-model mode)",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Short name for directory naming (default: last part of --model)",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help=(
            "Model architecture family for nnsight layer access. "
            f"Options: {list(ARCH_ALIASES.keys())}. "
            "Inferred from --model if omitted."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help=f"Dataset(s) to process (default: all). Available: {ALL_DATASETS}",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layer indices to extract activations from (required in single-model mode)",
    )
    parser.add_argument(
        "--noperiod",
        action="store_true",
        default=False,
        help="Strip trailing periods from statements (matches original --noperiod flag)",
    )
    parser.add_argument(
        "--acts_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/acts_original",
        help="Root directory for saved activations",
    )
    parser.add_argument(
        "--pca_dir",
        default="/storage/home/hcoda1/7/eayesh3/scratch/geometry_of_truth/pca_original",
        help="Root directory for PCA result .pt files",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures/pca_original",
        help="Root directory for PNG figures",
    )
    parser.add_argument(
        "--n_pcs",
        type=int,
        default=10,
        help="Number of PCs to compute (default: 10)",
    )
    return parser.parse_args()


def run_one(
    hf_id: str,
    model_name: str,
    arch: str,
    datasets: list[str],
    layers: list[int],
    batch_size: int,
    noperiod: bool,
    acts_dir: str,
    pca_dir: str,
    figures_dir: str,
    n_pcs: int,
) -> None:
    print(f"{'=' * 60}")
    print(f"Model: {hf_id}  (name: {model_name}, arch: {arch})")
    print(f"Layers: {layers}  |  batch_size: {batch_size}")
    print(f"{'=' * 60}")

    # Step 1: extract activations with nnsight
    print("Step 1: Extracting activations (nnsight)")
    try:
        extract_for_model(
            hf_id=hf_id,
            model_name=model_name,
            arch=arch,
            datasets=datasets,
            layers=layers,
            acts_dir=acts_dir,
            noperiod=noperiod,
        )
    except Exception as e:
        print(f"  ERROR during extraction: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        print()
        return

    # Step 2: PCA + figures (CPU, original get_pcs)
    print("\nStep 2: PCA (original get_pcs) + figures")
    try:
        run_pca_for_model(
            model_name=model_name,
            display_name=hf_id,
            datasets=datasets,
            layers=layers,
            acts_dir=acts_dir,
            pca_dir=pca_dir,
            figures_dir=figures_dir,
            n_pcs=n_pcs,
        )
    except Exception as e:
        print(f"  ERROR during PCA: {type(e).__name__}: {e}")

    print()


def main():
    args = parse_args()

    if args.config is not None:
        models, datasets, noperiod = load_config(args.config)
        # CLI --noperiod overrides config value
        noperiod = noperiod or args.noperiod

        print(f"Config:    {args.config}")
        print(f"Models:    {list(models.keys())}")
        print(f"Datasets:  {datasets}")
        print(f"noperiod:  {noperiod}")
        print(f"acts_dir:  {args.acts_dir}")
        print()

        for model_name, mcfg in models.items():
            run_one(
                hf_id=mcfg["hf_id"],
                model_name=model_name,
                arch=mcfg["arch"],
                datasets=datasets,
                layers=mcfg["layers"],
                batch_size=mcfg["batch_size"],
                noperiod=noperiod,
                acts_dir=args.acts_dir,
                pca_dir=args.pca_dir,
                figures_dir=args.figures_dir,
                n_pcs=args.n_pcs,
            )

    else:
        if args.model is None or args.layers is None:
            print("ERROR: provide either --config or both --model and --layers.")
            sys.exit(1)

        unknown_ds = [d for d in args.datasets if d not in ALL_DATASETS]
        if unknown_ds:
            raise ValueError(f"Unknown dataset(s): {unknown_ds}. Available: {ALL_DATASETS}")

        model_name = args.model_name or args.model.split("/")[-1].lower()
        arch = resolve_arch(args.arch) if args.arch else infer_arch(args.model)

        print(f"Datasets:  {args.datasets}")
        print(f"noperiod:  {args.noperiod}")
        print(f"acts_dir:  {args.acts_dir}")
        print()

        run_one(
            hf_id=args.model,
            model_name=model_name,
            arch=arch,
            datasets=args.datasets,
            layers=args.layers,
            batch_size=BATCH_SIZE,
            noperiod=args.noperiod,
            acts_dir=args.acts_dir,
            pca_dir=args.pca_dir,
            figures_dir=args.figures_dir,
            n_pcs=args.n_pcs,
        )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
