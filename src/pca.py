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


def run_pca(acts: torch.Tensor, k: int = 2) -> PCAResult:
    """
    Center, unit-variance scale, then run PCA on activations.

    Args:
        acts: [n_statements, hidden_size] float tensor (need not be pre-centered)
        k:    number of principal components to return

    Returns:
        PCAResult with projections, components, and explained variance ratios
    """
    acts = acts.float()
    acts = acts - acts.mean(dim=0)
    std = acts.std(dim=0)
    std[std < 1e-8] = 1.0
    acts = acts / std

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
    )


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
