"""
Visualization utilities for activation patching results.

Produces heatmaps matching Figure 2 of Marks & Tegmark (2024):
  - X-axis: token positions from the first differing token to end of prompt
             Differing tokens are labelled baseline_token/corrupted_token
  - Y-axis: layers (0 at top, matching the paper)
  - Color:  log P(baseline_answer) / P(corrupted_answer) — raw logit difference
"""

import numpy as np
import plotly.graph_objects as go


def _get_token_labels(result: dict, tokenizer=None) -> list[str]:
    """
    Return human-readable x-axis labels for the n_toks positions
    (index 0 = first differing token, index n_toks-1 = last token).

    Tokens that differ between baseline and corrupted are shown as
    "baseline_token/corrupted_token". Identical tokens show the shared text.

    Falls back to position indices if no tokenizer is provided.
    """
    n_toks = result["n_toks"]
    if tokenizer is None:
        return [f"pos -{n_toks - i}" for i in range(n_toks)]

    baseline_ids = tokenizer.encode(result["baseline_prompt"])[-n_toks:]
    corrupted_ids = tokenizer.encode(result["corrupted_prompt"])[-n_toks:]

    labels = []
    for b_id, c_id in zip(baseline_ids, corrupted_ids):
        b_str = tokenizer.decode([b_id]).strip() or repr(tokenizer.decode([b_id]))
        c_str = tokenizer.decode([c_id]).strip() or repr(tokenizer.decode([c_id]))
        if b_id != c_id:
            labels.append(f"{b_str}/{c_str}")
        else:
            labels.append(b_str)
    return labels



def _compute_nie(result: dict) -> np.ndarray:
    """
    Normalize logit diffs to [0, 1].

    baseline = FALSE prompt (floor, NIE=0)
    corrupted = TRUE prompt (ceiling, NIE=1)

    NIE = (logit_diff - baseline_logit_diff) / (corrupted_logit_diff - baseline_logit_diff)
    """
    logit_diffs = np.array(result["logit_diffs"], dtype=float)
    baseline = result["baseline_logit_diff"]
    corrupted = result["corrupted_logit_diff"]
    denom = corrupted - baseline
    if abs(denom) < 1e-8:
        return np.zeros_like(logit_diffs)
    return (logit_diffs - baseline) / denom


def plot_patching_heatmap(
    result: dict,
    tokenizer=None,
    title: str | None = None,
) -> go.Figure:
    """
    Build a Plotly heatmap for a single patching result.

    Color encodes the normalized indirect effect (0 = corrupted, 1 = baseline),
    with the actual logit-diff floor/ceiling shown below the title.

    Args:
        result:    One entry from `experimental_outputs/patching_*.json`.
        tokenizer: Optional HF tokenizer for human-readable x-axis labels.
        title:     Figure title; auto-generated from result if None.

    Returns:
        A Plotly Figure object (call `.show()` or `.write_image()`).
    """
    n_toks = result["n_toks"]
    n_layers = len(result["logit_diffs"][0])

    matrix = _compute_nie(result)  # [n_toks, n_layers], range ~[0, 1]
    zmin, zmax = 0.0, 1.0

    # matrix[0] = first differing token; transpose to [n_layers, n_toks] for Plotly
    z = matrix.T

    x_labels = _get_token_labels(result, tokenizer)
    y_labels = [str(layer) for layer in range(n_layers)]

    baseline_ld = result["baseline_logit_diff"]
    corrupted_ld = result["corrupted_logit_diff"]
    floor_ceil = (
        f"ceiling (baseline): {baseline_ld:.3f} | "
        f"floor (corrupted): {corrupted_ld:.3f}"
    )

    if title is None:
        model = result.get("model", "unknown")
        snippet = result["baseline_prompt"].split("\n")[-1][:60]
        title = f"<b>{model}</b><br><sup>{snippet}…<br>{floor_ceil}</sup>"
    else:
        title = f"{title}<br><sup>{floor_ceil}</sup>"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            zmin=zmin,
            zmax=zmax,
            colorscale="RdBu",
            reversescale=True,   # red = high NIE (baseline restored)
            colorbar=dict(
                title=dict(text="<b>NIE</b>", font=dict(size=18)),
                tickfont=dict(size=16),
            ),
            hoverongaps=False,
            hovertemplate="Layer %{y}, token %{x}<br>NIE: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis=dict(
            title=dict(
                text="<b>Token position</b>",
                font=dict(size=18),
            ),
            side="bottom",
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title=dict(
                text="<b>Layer</b>",
                font=dict(size=18),
            ),
            autorange="reversed",  # layer 0 at top
            tickfont=dict(size=16),
        ),
        font=dict(size=16),
        width=max(650, 90 * n_toks + 220),
        height=max(550, 22 * n_layers + 180),
        template="plotly_white",
    )

    return fig


def plot_all_patching_results(
    results: list[dict],
    tokenizer=None,
    max_plots: int | None = None,
) -> list[go.Figure]:
    """
    Build one heatmap figure per result entry.

    Args:
        results:   List of result dicts from `experimental_outputs/patching_*.json`.
        tokenizer: Optional HF tokenizer for x-axis token labels.
        max_plots: If set, limit to the first N results.

    Returns:
        List of Plotly Figure objects, one per result.
    """
    if max_plots is not None:
        results = results[:max_plots]
    return [plot_patching_heatmap(r, tokenizer=tokenizer) for r in results]
