"""
PCA utilities for visualizing truth representations.
"""

from dataclasses import dataclass

import numpy as np
import torch
import plotly.graph_objects as go


@dataclass
class PCAResult:
    projections: torch.Tensor    # [n_statements, k]
    components: torch.Tensor     # [hidden_size, k]
    explained_var_ratio: list[float]  # length k
    mean: torch.Tensor | None = None  # [hidden_size] mean used for centering
    std: torch.Tensor | None = None   # [hidden_size] std used for scaling


def run_pca(
    acts: torch.Tensor,
    k: int = 2,
    center: bool = True,
    scale: bool = False,
) -> PCAResult:
    """
    Optionally center and/or unit-variance scale, then run PCA on activations.

    Args:
        acts:   [n_statements, hidden_size] float tensor
        k:      number of principal components to return
        center: subtract per-dimension mean before PCA (default: True)
        scale:  divide by per-dimension std before PCA (default: True)

    Returns:
        PCAResult with projections, components, and explained variance ratios
    """
    acts = acts.float()
    saved_mean = None
    saved_std = None
    if center:
        saved_mean = acts.mean(dim=0)
        acts = acts - saved_mean
    if scale:
        saved_std = acts.std(dim=0)
        saved_std[saved_std < 1e-8] = 1.0
        acts = acts / saved_std

    cov = torch.mm(acts.T, acts) / (acts.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # eigh returns ascending order — reverse to get descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    explained_var_ratio = (eigenvalues[:k] / eigenvalues.sum()).tolist()
    components = eigenvectors[:, :k]
    projections = torch.mm(acts, components)

    return PCAResult(
        projections=projections,
        components=components,
        explained_var_ratio=explained_var_ratio,
        mean=saved_mean,
        std=saved_std,
    )


def project_probe(
    direction: torch.Tensor,
    result: PCAResult,
    acts_mean: torch.Tensor | None = None,
    acts_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Project a probe direction vector from activation space into the 2D PC space
    defined by a PCAResult.

    If the PCA was computed with centering and/or scaling (the default), pass
    acts_mean and acts_std so the direction is transformed into the same
    preprocessed space before projection. Without them, the direction is
    projected directly onto the components (correct if the probe was trained
    on already-preprocessed activations, or if centering/scaling can be ignored
    for the direction).

    Args:
        direction: [hidden_size] unit probe direction in activation space
        result:    PCAResult from run_pca
        acts_mean: [hidden_size] mean used during PCA preprocessing (optional)
        acts_std:  [hidden_size] std used during PCA preprocessing (optional)

    Returns:
        [2] tensor — the probe direction expressed in PC1/PC2 coordinates
    """
    d = direction.float()
    # Transform direction into the preprocessed space.
    # Centering shifts the origin but does not rotate directions, so it has no
    # effect on a direction vector. Scaling does affect directions.
    if acts_std is not None:
        d = d * acts_std  # direction in scaled space: w_scaled = w * std
    return result.components.T @ d  # [2]


def plot_pca_with_probe(
    result: PCAResult,
    labels: list[int],
    probe,
    title: str = "",
    acts_mean: torch.Tensor | None = None,
    acts_std: torch.Tensor | None = None,
) -> go.Figure:
    """
    PCA scatter plot with the probe direction overlaid as an arrow and the
    decision boundary drawn as a line.

    Works with both LRProbe and MassMeanProbe (anything with a .direction
    attribute and a ._bias float, or an ._lr with .intercept_).

    Args:
        result:    PCAResult from run_pca
        labels:    list of 0/1 labels
        probe:     fitted LRProbe or MassMeanProbe
        title:     figure title
        acts_mean: [hidden_size] mean used during PCA (needed if scale=True was used)
        acts_std:  [hidden_size] std used during PCA (needed if scale=True was used)

    Returns:
        Plotly Figure
    """
    proj = result.projections.cpu().numpy()
    ev = result.explained_var_ratio
    labels_arr = np.array(labels)

    # ── Data scatter ──────────────────────────────────────────────────────────
    traces = []
    for val, name, color in [(1, "True", "#d62728"), (0, "False", "#1f77b4")]:
        mask = labels_arr == val
        traces.append(
            go.Scatter(
                x=proj[mask, 0],
                y=proj[mask, 1],
                mode="markers",
                name=name,
                marker=dict(color=color, size=5, opacity=0.7),
            )
        )

    # ── Probe direction arrow ─────────────────────────────────────────────────
    d_pc = project_probe(probe.direction, result, acts_mean, acts_std).cpu().numpy()
    # Scale arrow to 40% of the plot range for visibility
    x_range = proj[:, 0].ptp()
    y_range = proj[:, 1].ptp()
    scale = 0.4 * max(x_range, y_range) / (np.linalg.norm(d_pc) + 1e-8)
    cx, cy = proj[:, 0].mean(), proj[:, 1].mean()  # centre on data centroid

    traces.append(go.Scatter(
        x=[cx, cx + d_pc[0] * scale],
        y=[cy, cy + d_pc[1] * scale],
        mode="lines+markers",
        name="Probe direction",
        line=dict(color="#2ca02c", width=3),
        marker=dict(symbol="arrow", size=12, angleref="previous", color="#2ca02c"),
        showlegend=True,
    ))

    # ── Decision boundary ─────────────────────────────────────────────────────
    # Boundary in PC space: d_pc · z = bias_pc
    # For MassMeanProbe: bias comes from ._bias (in original activation space,
    #   but since direction is unit-norm and we project consistently, we use the
    #   projection of the boundary offset onto the PC direction).
    # For LRProbe: bias is -intercept / ||w|| projected similarly.
    #
    # Simplest correct approach: the boundary passes through the point
    #   z_boundary = bias_pc / ||d_pc||^2 * d_pc
    # and is perpendicular to d_pc in PC space.
    from src.probes import LRProbe, MassMeanProbe
    if isinstance(probe, LRProbe):
        w = torch.tensor(probe._lr.coef_[0], dtype=torch.float32)
        b = float(probe._lr.intercept_[0])
        if acts_std is not None:
            w_scaled = w * acts_std.float()
        else:
            w_scaled = w
        d_pc_raw = (result.components.T @ w_scaled).cpu().numpy()
        # boundary: d_pc_raw · z + b = 0  →  z · d_pc_raw = -b
        bias_pc = -b
    else:  # MassMeanProbe
        d_pc_raw = d_pc  # already in PC space
        bias_pc = probe._bias

    # Draw boundary as a line perpendicular to d_pc_raw across the plot range
    norm_sq = float(np.dot(d_pc_raw, d_pc_raw)) + 1e-12
    # Point on boundary closest to origin
    p0 = d_pc_raw * (bias_pc / norm_sq)
    # Perpendicular direction (rotate 90°)
    perp = np.array([-d_pc_raw[1], d_pc_raw[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-8)
    half = 0.6 * max(x_range, y_range)

    traces.append(go.Scatter(
        x=[p0[0] - perp[0] * half, p0[0] + perp[0] * half],
        y=[p0[1] - perp[1] * half, p0[1] + perp[1] * half],
        mode="lines",
        name="Decision boundary",
        line=dict(color="#2ca02c", width=2, dash="dash"),
    ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=f"PC1 ({ev[0]:.1%} var)",
        yaxis_title=f"PC2 ({ev[1]:.1%} var)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        legend=dict(font=dict(size=14)),
    )
    return fig


def project_onto_pca(
    target_acts: torch.Tensor,
    source_result: PCAResult,
) -> torch.Tensor:
    """
    Project target_acts into the PC space defined by source_result, applying
    the same preprocessing (centering and scaling) that was used when the PCA
    was computed.

    Args:
        target_acts:   [n_target, hidden_size] activations to project
        source_result: PCAResult from run_pca on the source dataset

    Returns:
        [n_target, k] projections of target_acts in the source PC space
    """
    acts = target_acts.float()
    if source_result.mean is not None:
        acts = acts - source_result.mean
    if source_result.std is not None:
        acts = acts / source_result.std
    return torch.mm(acts, source_result.components)


def cross_dataset_pca(
    source_dataset: str,
    target_dataset: str,
    model_name: str,
    layer: int,
    acts_dir: str = "acts",
    k: int = 2,
    center: bool = True,
    scale: bool = True,
) -> tuple[PCAResult, torch.Tensor, list[int], list[int]]:
    """
    Fit PCA on source_dataset activations and project target_dataset into
    that same PC space.

    Args:
        source_dataset: dataset whose activations define the PC axes
        target_dataset: dataset to project into source PC space
        model_name:     model key used for loading activations
        layer:          layer index
        acts_dir:       root directory for activations
        k:              number of PCs
        center:         mean-center before PCA (using source mean)
        scale:          std-scale before PCA (using source std)

    Returns:
        source_result:      PCAResult for source_dataset (defines the space)
        target_projections: [n_target, k] projections of target_dataset
        source_labels:      labels for source_dataset
        target_labels:      labels for target_dataset
    """
    import os
    from src.activations import load_acts as _load_acts

    source_acts = _load_acts(model_name, source_dataset, layer, acts_dir, center=False)
    target_acts = _load_acts(model_name, target_dataset, layer, acts_dir, center=False)

    source_result = run_pca(source_acts, k=k, center=center, scale=scale)
    target_projections = project_onto_pca(target_acts, source_result)

    def _load_labels(dataset_name):
        labels_path = os.path.join(acts_dir, model_name, dataset_name, "labels.pt")
        if os.path.exists(labels_path):
            return torch.load(labels_path, weights_only=True).tolist()
        from src.data import load_dataset
        _, labels = load_dataset(dataset_name)
        return labels

    return source_result, target_projections, _load_labels(source_dataset), _load_labels(target_dataset)


def plot_cross_dataset_pca(
    source_result: PCAResult,
    target_projections: torch.Tensor,
    source_labels: list[int],
    target_labels: list[int],
    source_name: str,
    target_name: str,
    title: str = "",
) -> go.Figure:
    """
    Plot source dataset in its own PC space alongside target dataset projected
    into that same space. Source points are circles, target points are crosses.

    Args:
        source_result:      PCAResult from run_pca on the source dataset
        target_projections: [n_target, k] from project_onto_pca or cross_dataset_pca
        source_labels:      0/1 labels for source
        target_labels:      0/1 labels for target
        source_name:        legend label for source dataset
        target_name:        legend label for target dataset
        title:              figure title
    """
    src_proj = source_result.projections.cpu().numpy()
    tgt_proj = target_projections.cpu().numpy()
    ev = source_result.explained_var_ratio

    src_labels = np.array(source_labels)
    tgt_labels = np.array(target_labels)

    traces = []
    for val, true_false, src_color, tgt_color in [
        (1, "True",  "#d62728", "#ff7f7f"),
        (0, "False", "#1f77b4", "#7fbfff"),
    ]:
        mask = src_labels == val
        traces.append(go.Scatter(
            x=src_proj[mask, 0], y=src_proj[mask, 1],
            mode="markers",
            name=f"{source_name} — {true_false}",
            marker=dict(color=src_color, size=5, opacity=0.7, symbol="circle"),
        ))
        mask = tgt_labels == val
        traces.append(go.Scatter(
            x=tgt_proj[mask, 0], y=tgt_proj[mask, 1],
            mode="markers",
            name=f"{target_name} — {true_false}",
            marker=dict(color=tgt_color, size=6, opacity=0.7, symbol="x"),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=f"PC1 ({ev[0]:.1%} var, {source_name})",
        yaxis_title=f"PC2 ({ev[1]:.1%} var, {source_name})",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        legend=dict(font=dict(size=14)),
    )
    return fig


def plot_pca(
    result: PCAResult,
    labels: list[int],
    title: str = "",
) -> go.Figure:
    """
    2D scatter plot of the first two PCs, colored by true (1) / false (0) label.

    Args:
        result: output of run_pca
        labels: list of 0/1 labels, length n_statements
        title:  figure title

    Returns:
        Plotly Figure
    """
    proj = result.projections.cpu().numpy()
    ev = result.explained_var_ratio
    labels_arr = np.array(labels)

    traces = []
    for val, name, color in [(1, "True", "#d62728"), (0, "False", "#1f77b4")]:
        mask = labels_arr == val
        traces.append(
            go.Scatter(
                x=proj[mask, 0],
                y=proj[mask, 1],
                mode="markers",
                name=name,
                marker=dict(color=color, size=5, opacity=0.7),
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=f"PC1 ({ev[0]:.1%} var)",
        yaxis_title=f"PC2 ({ev[1]:.1%} var)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        legend=dict(font=dict(size=14)),
    )
    return fig
