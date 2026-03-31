"""
Shared plotting utilities for probe summary visualizations.

Provides:
  load_probe_scores     — flexible results.json loader (auto-detects probe columns;
                          works for both regular probes (mm_cov) and original probes
                          (mm_iid, ccs)).
  plot_probe_summary    — bar chart of mean accuracy over test datasets, with SE
                          error bars and individual test-dataset score points.
  plot_transfer_summary — grouped bar chart comparing probe transfer across source
                          models vs. in-domain baseline, with two-level x-axis
                          (outer: train datasets or layers; inner: probe types).
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── Display names and colors for known probe types ────────────────────────────

PROBE_DISPLAY: dict[str, str] = {
    "lr":     "LR",
    "mm":     "MM",
    "mm_cov": "MM+cov",
    "mm_iid": "MM+iid",
    "ccs":    "CCS",
}

PROBE_COLORS: dict[str, str] = {
    "lr":     "#1f77b4",  # blue
    "mm":     "#ff7f0e",  # orange
    "mm_cov": "#2ca02c",  # green
    "mm_iid": "#2ca02c",  # green  (same role as mm_cov)
    "ccs":    "#d62728",  # red
}

# Distinct palette for probe source models in transfer plots
SOURCE_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
]

# Muted palette for individual test-dataset points/lines
_DS_PALETTE = [
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#17becf", "#bcbd22",
]

# Canonical ordering for probe columns (used in detect_probe_cols)
_PROBE_ORDER = ["lr", "mm", "mm_cov", "mm_iid"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_probe_scores(results_path: str) -> pd.DataFrame:
    """
    Load a results.json file into a DataFrame.

    Unlike src.probes.load_probe_results, this version auto-detects probe
    columns rather than hardcoding "mm_cov", so it works for both regular
    probes (lr, mm, mm_cov) and original probes (lr, mm, mm_iid, ccs).

    Returns:
        DataFrame with MultiIndex (train_group, test_dataset) and one column
        per probe type present in the file, plus an "n" column.
    """
    with open(results_path) as f:
        raw = json.load(f)

    rows = []
    for train_key, info in raw.items():
        for test_ds, scores in info["scores"].items():
            row = {"train_group": train_key, "test_dataset": test_ds}
            for k, v in scores.items():
                row[k] = int(v) if k == "n" else float(v)
            rows.append(row)

    return pd.DataFrame(rows).set_index(["train_group", "test_dataset"])


def detect_probe_cols(df: pd.DataFrame) -> list[str]:
    """Return probe columns present in df, in canonical order (excludes 'n')."""
    return [c for c in _PROBE_ORDER if c in df.columns]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_probe_summary(
    data: dict[str, pd.DataFrame],
    probe_cols: list[str],
    title: str = "",
) -> go.Figure:
    """
    Bar chart with mean accuracy over test datasets, SE error bars,
    and individual test-dataset score points (single color, no lines).
    Legend is placed horizontally below the figure.

    Args:
        data:       Ordered dict mapping x-axis label -> DataFrame with
                    index = test_dataset, columns ⊇ probe_cols.
                    Each entry represents one x-axis group (e.g. a train
                    group name or "layer 15").
        probe_cols: Probe type columns to plot, e.g. ["lr", "mm", "mm_cov"].
                    Bars are color-coded per probe type.
        title:      Figure title.

    Returns:
        Plotly Figure.
    """
    x_labels      = list(data.keys())
    n_groups      = len(x_labels)
    n_probes      = len(probe_cols)
    group_width   = 0.7
    bar_width     = group_width / n_probes
    group_centers = np.arange(n_groups, dtype=float)
    offsets       = np.linspace(
        -(group_width - bar_width) / 2,
         (group_width - bar_width) / 2,
        n_probes,
    )

    # Collect all test datasets in stable order
    seen: set[str] = set()
    all_test_datasets: list[str] = []
    for df in data.values():
        for ds in df.index:
            if ds not in seen:
                all_test_datasets.append(ds)
                seen.add(ds)

    fig = go.Figure()

    for pi, probe in enumerate(probe_cols):
        bar_x = group_centers + offsets[pi]
        color = PROBE_COLORS.get(probe, "#7f7f7f")
        label = PROBE_DISPLAY.get(probe, probe)

        means, ses = [], []
        for xl in x_labels:
            df = data[xl]
            if probe in df.columns:
                vals = df[probe].dropna()
                n    = len(vals)
                means.append(float(vals.mean()))
                ses.append(float(vals.std() / np.sqrt(n)) if n > 1 else 0.0)
            else:
                means.append(None)
                ses.append(None)

        # ── Bar trace (mean ± SE) ─────────────────────────────────────────────
        fig.add_trace(go.Bar(
            x=bar_x.tolist(),
            y=means,
            error_y=dict(
                type="data",
                array=ses,
                visible=True,
                thickness=1.5,
                width=4,
                color="rgba(0,0,0,0.45)",
            ),
            width=bar_width * 0.85,
            name=label,
            marker_color=color,
            legendgroup=f"probe_{probe}",
            showlegend=True,
        ))

        # ── Per-test-dataset scatter (single color, no lines) ─────────────────
        for ds in all_test_datasets:
            xs, ys = [], []
            for j, xl in enumerate(x_labels):
                df = data[xl]
                if probe in df.columns and ds in df.index:
                    xs.append(float(bar_x[j]))
                    ys.append(float(df.loc[ds, probe]))

            if not ys:
                continue

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=5, color="rgba(30,30,30,0.45)"),
                showlegend=False,
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            tickvals=group_centers.tolist(),
            ticktext=x_labels,
            tickangle=-30,
            tickfont=dict(size=11),
        ),
        yaxis=dict(title="Accuracy", range=[0, 1], tickformat=".0%"),
        shapes=[dict(
            type="line", x0=0, x1=1, y0=0.5, y1=0.5,
            xref="paper", yref="y",
            line=dict(color="red", width=1.5, dash="dash"),
        )],
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.3,
            xanchor="center",
            yanchor="top",
            font=dict(size=11),
        ),
        template="plotly_white",
        height=520,
        margin=dict(b=160, r=40, t=60),
    )
    return fig


def plot_transfer_summary(
    data: dict[str, dict[str, pd.DataFrame]],
    probe_cols: list[str],
    source_labels: list[str],
    source_colors: dict[str, str],
    title: str = "",
) -> go.Figure:
    """
    Grouped bar chart comparing probe transfer across source models vs. in-domain.

    Two-level x-axis:
      Outer groups (annotated below ticks): train datasets or layers.
      Inner groups (tick labels):           probe types (LR, MM, MM+iid, ...).

    Within each inner group, one bar per source model (color-coded).
    Bars show mean ± SE over test datasets; individual scores shown as dark dots.
    Dashed vertical lines separate outer groups.

    Args:
        data:          outer_label -> source_label -> DataFrame with index =
                       test_dataset and columns ⊇ probe_cols.
                       Missing outer/source keys are silently skipped (empty bar).
        probe_cols:    Probe columns to plot in order, e.g. ["lr", "mm", "mm_cov"].
        source_labels: All source labels in display order (for consistent positioning
                       even when some sources are absent for a given outer group).
        source_colors: Maps source_label -> hex color string.
        title:         Figure title.
    """
    outer_labels  = list(data.keys())
    n_outer       = len(outer_labels)
    n_inner       = len(probe_cols)
    n_src         = len(source_labels)

    total_width   = 0.85
    inner_gap     = 0.05
    inner_grp_w   = (total_width - inner_gap * (n_inner - 1)) / n_inner
    bar_width     = inner_grp_w / n_src * 0.85
    outer_centers = np.arange(n_outer, dtype=float)

    def _inner_center(k: int, i: int) -> float:
        return float(outer_centers[k] + (i - (n_inner - 1) / 2) * (inner_grp_w + inner_gap))

    def _bar_center(k: int, i: int, j: int) -> float:
        return _inner_center(k, i) + (j - (n_src - 1) / 2) * (inner_grp_w / n_src)

    fig = go.Figure()

    for j, src_label in enumerate(source_labels):
        color = source_colors.get(src_label, "#7f7f7f")
        bar_xs, bar_ys, bar_ses = [], [], []
        dot_xs, dot_ys = [], []

        for k, outer in enumerate(outer_labels):
            for i, probe in enumerate(probe_cols):
                bx = _bar_center(k, i, j)
                df = data.get(outer, {}).get(src_label)

                if df is not None and probe in df.columns:
                    vals = df[probe].dropna()
                    n    = len(vals)
                    bar_ys.append(float(vals.mean()))
                    bar_ses.append(float(vals.std() / np.sqrt(n)) if n > 1 else 0.0)
                    dot_xs.extend([bx] * n)
                    dot_ys.extend(vals.tolist())
                else:
                    bar_ys.append(None)
                    bar_ses.append(0.0)

                bar_xs.append(bx)

        fig.add_trace(go.Bar(
            x=bar_xs,
            y=bar_ys,
            error_y=dict(
                type="data",
                array=bar_ses,
                visible=True,
                thickness=1.5,
                width=3,
                color="rgba(0,0,0,0.4)",
            ),
            width=bar_width * 0.9,
            name=src_label,
            marker_color=color,
            showlegend=True,
        ))

        if dot_xs:
            fig.add_trace(go.Scatter(
                x=dot_xs,
                y=dot_ys,
                mode="markers",
                marker=dict(size=4, color="rgba(30,30,30,0.4)"),
                showlegend=False,
            ))

    # ── Separator lines between outer groups ──────────────────────────────────
    sep_shapes = [
        dict(
            type="line",
            x0=(outer_centers[k] + outer_centers[k + 1]) / 2,
            x1=(outer_centers[k] + outer_centers[k + 1]) / 2,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="rgba(0,0,0,0.12)", width=1, dash="dash"),
        )
        for k in range(n_outer - 1)
    ]
    chance_shape = dict(
        type="line", x0=0, x1=1, y0=0.5, y1=0.5,
        xref="paper", yref="y",
        line=dict(color="red", width=1.5, dash="dash"),
    )

    if n_inner == 1:
        # Single probe type — use outer labels directly as x-axis ticks, no annotations
        tick_vals   = outer_centers.tolist()
        tick_texts  = outer_labels
        annotations = []
        legend_y    = -0.22
        bottom_margin = 150
    else:
        # Multiple probe types — two-level x-axis
        tick_vals  = [_inner_center(k, i) for k in range(n_outer) for i in range(n_inner)]
        tick_texts = [
            PROBE_DISPLAY.get(probe_cols[i], probe_cols[i])
            for k in range(n_outer)
            for i in range(n_inner)
        ]
        annotations = [
            dict(
                x=float(outer_centers[k]),
                y=-0.22,
                xref="x",
                yref="paper",
                text=f"<b>{outer_labels[k]}</b>",
                showarrow=False,
                font=dict(size=11),
                xanchor="center",
            )
            for k in range(n_outer)
        ]
        legend_y      = -0.38
        bottom_margin = 210

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickangle=-30,
            tickfont=dict(size=10),
        ),
        yaxis=dict(title="Accuracy", range=[0, 1], tickformat=".0%"),
        legend=dict(
            orientation="h",
            x=0.5,
            y=legend_y,
            xanchor="center",
            yanchor="top",
            font=dict(size=10),
        ),
        annotations=annotations,
        shapes=sep_shapes + [chance_shape],
        template="plotly_white",
        height=520,
        margin=dict(b=bottom_margin, r=40, t=60),
    )
    return fig
