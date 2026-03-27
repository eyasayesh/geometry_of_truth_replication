"""
Activation patching experiment (replicates Section 3 of the paper).

For a pair of prompts that differ starting at some token (baseline vs. corrupted):
  baseline = FALSE prompt  (the run we intervene on)
  corrupted = TRUE prompt  (the source of activations we patch in)

  1. Cache residual-stream activations from every layer of the corrupted prompt.
  2. For each token position (from diverging token to end) and each layer,
     run the baseline prompt with ONLY that one (layer, position) activation
     swapped in, and record logit(baseline_answer) - logit(corrupted_answer).

The result is a [n_toks x n_layers] matrix of logit differences (Figure 2 in
the paper), showing which specific (layer, position) hidden states causally
mediate the behavioural difference between the two prompts.

Each row (tok_idx) is computed in a single batched forward pass across all
layers, so total forward passes = O(n_toks), not O(n_toks * n_layers).
"""

import torch
from transformer_lens import HookedTransformer

from src.activations import resid_post_hook


def get_diverging_token_count(
    model: HookedTransformer,
    baseline_prompt: str,
    corrupted_prompt: str,
) -> int:
    """
    Return the number of token positions from the end (inclusive of the first
    differing token) that need to be patched. Raises if prompt lengths differ.
    """
    baseline_toks = model.to_tokens(baseline_prompt, prepend_bos=True)[0].tolist()
    corrupted_toks = model.to_tokens(corrupted_prompt, prepend_bos=True)[0].tolist()

    if len(baseline_toks) != len(corrupted_toks):
        raise ValueError(
            f"Baseline and corrupted prompts must tokenize to the same length, "
            f"got {len(baseline_toks)} vs {len(corrupted_toks)}."
        )

    same = [b == c for b, c in zip(baseline_toks, corrupted_toks)]
    # Find first difference from the end; n_toks covers that token and everything after
    n_toks = next(i + 1 for i, s in enumerate(reversed(same)) if not s)
    return n_toks


@torch.no_grad()
def run_patching_experiment(
    model: HookedTransformer,
    baseline_prompt: str,
    corrupted_prompt: str,
    baseline_answer: str,
    corrupted_answer: str,
) -> dict:
    """
    Run the full activation patching experiment across all layers and token
    positions, producing a [n_toks x n_layers] logit difference matrix.

    For each token position (outer loop, O(n_toks) forward passes), all layers
    are patched simultaneously in one batched forward pass: n_layers copies of
    the corrupted tokens are run, and the hook at layer L patches only batch
    item L at the current position, leaving all other items untouched.

    Args:
        baseline_prompt:  the clean/source prompt
        corrupted_prompt: the prompt we intervene on
        baseline_answer:  string token for the expected baseline output (e.g. " TRUE")
        corrupted_answer: string token for the expected corrupted output (e.g. " FALSE")

    Returns a dict with:
      - baseline_prompt, corrupted_prompt, baseline_answer, corrupted_answer
      - n_toks: number of token positions patched (from diverging token to end)
      - baseline_logit_diff:  logit diff on the unpatched baseline prompt  (ceiling for NIE)
      - corrupted_logit_diff: logit diff on the unpatched corrupted prompt (floor for NIE)
      - logit_diffs: list of lists [n_toks x n_layers]
                     logit_diffs[tok_idx][layer] = logit diff after patching
                     layer `layer` at position `-(n_toks - tok_idx)` only
                     (tok_idx=0 is the first differing token, tok_idx=n_toks-1 is the last token)
                     NIE = (logit_diffs[tok_idx][layer] - corrupted_logit_diff)
                           / (baseline_logit_diff - corrupted_logit_diff)
    """
    baseline_tokens = model.to_tokens(baseline_prompt, prepend_bos=True)
    corrupted_tokens = model.to_tokens(corrupted_prompt, prepend_bos=True)
    n_layers = model.cfg.n_layers

    baseline_tok_id = model.to_single_token(baseline_answer)
    corrupted_tok_id = model.to_single_token(corrupted_answer)

    n_toks = get_diverging_token_count(model, baseline_prompt, corrupted_prompt)

    # Step 1: cache all residual-stream activations from the corrupted prompt.
    # Also compute corrupted logit diff from the same forward pass.
    hook_names = [resid_post_hook(layer) for layer in range(n_layers)]
    corrupted_logits, corrupted_cache = model.run_with_cache(
        corrupted_tokens, names_filter=hook_names
    )
    corrupted_logit_diff = (
        corrupted_logits[0, -1, baseline_tok_id]
        - corrupted_logits[0, -1, corrupted_tok_id]
    ).item()

    # Logit diff on the unpatched baseline prompt
    baseline_logits = model(baseline_tokens)
    baseline_logit_diff = (
        baseline_logits[0, -1, baseline_tok_id]
        - baseline_logits[0, -1, corrupted_tok_id]
    ).item()

    # Step 2: for each token position, patch all layers in one batched forward pass.
    #
    # n_layers copies of the baseline tokens. Hook at layer L patches ONLY
    # batch item L at position `pos` with the cached corrupted activation,
    # leaving all other batch items untouched.
    # Each cell logit_diffs[tok_idx][layer] is the effect of patching exactly
    # one (layer, position) hidden state.
    logit_diffs = [[None] * n_layers for _ in range(n_toks)]

    batch_tokens = baseline_tokens.repeat(n_layers, 1)  # [n_layers, seq]

    # tok_idx=0 → first differing token (pos=-n_toks), tok_idx=n_toks-1 → last token (pos=-1)
    for tok_idx in range(n_toks):
        pos = -(n_toks - tok_idx)  # left-to-right: first differing token first

        def make_hook(layer_idx: int, patch: torch.Tensor, pos: int = pos):
            def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
                value[layer_idx, pos, :] = patch
                return value
            return hook_fn

        hooks = [
            (
                resid_post_hook(layer),
                make_hook(layer, corrupted_cache[resid_post_hook(layer)][0, pos, :]),
            )
            for layer in range(n_layers)
        ]

        patched_logits = model.run_with_hooks(batch_tokens, fwd_hooks=hooks)
        # patched_logits: [n_layers, seq, vocab]

        for layer in range(n_layers):
            logit_diffs[tok_idx][layer] = (
                patched_logits[layer, -1, baseline_tok_id]
                - patched_logits[layer, -1, corrupted_tok_id]
            ).item()

    return {
        "baseline_prompt": baseline_prompt,
        "corrupted_prompt": corrupted_prompt,
        "baseline_answer": baseline_answer,
        "corrupted_answer": corrupted_answer,
        "n_toks": n_toks,
        "baseline_logit_diff": baseline_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
        "logit_diffs": logit_diffs,  # [n_toks][n_layers]
    }
