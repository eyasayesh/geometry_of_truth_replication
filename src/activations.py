"""
Activation extraction and patching utilities built on TransformerLens.

TransformerLens hook points used:
  blocks.{layer}.hook_resid_post  — residual stream after full block (attn + MLP)

Save format mirrors geometry-of-truth:
  acts/{model_name}/{dataset_name}/layer_{layer}_{idx}.pt
Each file holds a [batch_size, hidden_size] float32 tensor, enabling
direct use of the original probing notebooks.
"""

import os
from glob import glob
from typing import Callable

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

BATCH_SIZE = 25  # matches geometry-of-truth format

# Hook point for the residual stream output of a given layer
def resid_post_hook(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


@torch.no_grad()
def extract_acts(
    model: HookedTransformer,
    statements: list[str],
    layers: list[int],
    batch_size: int = BATCH_SIZE,
) -> dict[int, torch.Tensor]:
    """
    Extract residual-stream activations at the final token position for each layer.

    Uses run_with_cache with a names_filter so only the requested layers are
    stored in memory during each forward pass.

    Returns:
        dict mapping layer_idx -> tensor of shape [n_statements, hidden_size]
    """
    hook_names = [resid_post_hook(layer) for layer in layers]
    all_acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(statements), batch_size), desc="Extracting activations"):
        batch = statements[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)  # left-pads via tokenizer

        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        # With left-padding, position -1 is always the last real token
        for layer in layers:
            act = cache[resid_post_hook(layer)][:, -1, :].cpu()
            all_acts[layer].append(act)

    return {layer: torch.cat(acts, dim=0) for layer, acts in all_acts.items()}


@torch.no_grad()
def patch_and_run(
    model: HookedTransformer,
    tokens: torch.Tensor,
    patch_tensor: torch.Tensor,
    layer: int,
    tok_idx: int = -1,
) -> torch.Tensor:
    """
    Run a forward pass on tokens, patching the residual stream at (layer, tok_idx)
    with patch_tensor before continuing downstream.

    Args:
        tokens:       [batch, seq] token ids
        patch_tensor: [batch, hidden] or [hidden] values to insert
        layer:        which transformer layer to patch after
        tok_idx:      which token position to patch (-1 = last)

    Returns:
        logits: [batch, seq, vocab]
    """
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        value[:, tok_idx, :] = patch_tensor
        return value

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(resid_post_hook(layer), hook_fn)],
    )
    return logits


def save_acts(
    acts: dict[int, torch.Tensor],
    model_name: str,
    dataset_name: str,
    output_dir: str = "acts",
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Save activations in batched .pt files matching geometry-of-truth format:
      {output_dir}/{model_name}/{dataset_name}/layer_{layer}_{idx}.pt
    """
    save_dir = os.path.join(output_dir, model_name, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    for layer, tensor in acts.items():
        for idx in range(0, tensor.shape[0], batch_size):
            chunk = tensor[idx : idx + batch_size]
            torch.save(chunk, os.path.join(save_dir, f"layer_{layer}_{idx}.pt"))


def load_acts(
    model_name: str,
    dataset_name: str,
    layer: int,
    output_dir: str = "acts",
    center: bool = True,
) -> torch.Tensor:
    """
    Load all saved activation chunks for a given model/dataset/layer and
    concatenate into a single [n_statements, hidden_size] float32 tensor.
    """
    directory = os.path.join(output_dir, model_name, dataset_name)
    files = sorted(
        glob(os.path.join(directory, f"layer_{layer}_*.pt")),
        key=lambda f: int(os.path.basename(f).split("_")[-1].replace(".pt", "")),
    )
    if not files:
        raise FileNotFoundError(
            f"No activations found at '{directory}' for layer {layer}. "
            "Run generate_acts.py first."
        )
    acts = torch.cat(
        [torch.load(f, weights_only=True) for f in files], dim=0
    ).float()
    if center:
        acts = acts - acts.mean(dim=0)
    return acts
