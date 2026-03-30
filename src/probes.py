"""
Linear probes for truth direction extraction (Section 5 of the paper).

Two probe types:
  LRProbe       — logistic regression (sklearn).
  MassMeanProbe — difference-in-means; optionally with within-class
                  covariance correction (equivalent to LDA direction).

Both expose:
  .direction  — unit vector in activation space (for causal interventions)
  .fit()      — train on (acts, labels)
  .predict()  — binary predictions
  .score()    — accuracy on (acts, labels)
  .save()/.load() — serialise/deserialise probe state
"""

import torch
import numpy as np
from torch import Tensor
from sklearn.linear_model import LogisticRegression


class LRProbe:
    """Logistic regression probe trained with sklearn."""

    def __init__(self):
        self._lr: LogisticRegression | None = None
        self.direction: Tensor | None = None  # unit-norm weight vector

    def fit(self, acts: Tensor, labels: Tensor) -> "LRProbe":
        X = acts.float().numpy()
        y = labels.numpy()
        self._lr = LogisticRegression(max_iter=1000, fit_intercept=True, C=0.1)
        self._lr.fit(X, y)
        w = torch.tensor(self._lr.coef_[0], dtype=torch.float32)
        self.direction = w / w.norm()
        return self

    def predict(self, acts: Tensor) -> Tensor:
        return torch.tensor(
            self._lr.predict(acts.float().numpy()), dtype=torch.long
        )

    def predict_proba(self, acts: Tensor) -> Tensor:
        """Returns P(label=1) for each sample."""
        proba = self._lr.predict_proba(acts.float().numpy())
        return torch.tensor(proba[:, 1], dtype=torch.float32)

    def score(self, acts: Tensor, labels: Tensor) -> float:
        return float(self._lr.score(acts.float().numpy(), labels.numpy()))

    def save(self, path: str) -> None:
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"type": "lr", "lr": pickle.dumps(self._lr), "direction": self.direction}, path)

    @classmethod
    def load(cls, path: str) -> "LRProbe":
        import pickle
        state = torch.load(path, weights_only=False)
        probe = cls()
        probe._lr = pickle.loads(state["lr"])
        probe.direction = state["direction"]
        return probe


class MassMeanProbe:
    """
    Difference-in-means probe.

      θ_mm  = μ⁺ - μ⁻
      θ_cov = Σ_w⁻¹ θ_mm   (within-class covariance correction, i.e. LDA)

    The prediction threshold is the midpoint of the two class mean projections.
    """

    def __init__(self):
        self.direction: Tensor | None = None
        self._bias: float = 0.0
        self._use_cov: bool = False

    def fit(self, acts: Tensor, labels: Tensor, use_cov: bool = False) -> "MassMeanProbe":
        acts = acts.float()
        labels = labels.long()
        self._use_cov = use_cov

        pos = acts[labels == 1]
        neg = acts[labels == 0]

        mu_pos = pos.mean(dim=0)
        mu_neg = neg.mean(dim=0)
        theta = mu_pos - mu_neg

        if use_cov:
            # Pooled within-class covariance
            cov = (torch.cov(pos.T) + torch.cov(neg.T)) / 2
            cov += 1e-6 * torch.eye(cov.shape[0])
            theta = torch.linalg.solve(cov, theta.unsqueeze(-1)).squeeze(-1)

        self.direction = theta / theta.norm()
        self._bias = float((self.direction @ mu_pos + self.direction @ mu_neg) / 2)
        return self

    def _project(self, acts: Tensor) -> Tensor:
        return acts.float() @ self.direction

    def predict(self, acts: Tensor) -> Tensor:
        return (self._project(acts) > self._bias).long()

    def predict_proba(self, acts: Tensor) -> Tensor:
        """Soft score via sigmoid of signed distance from the decision boundary."""
        return torch.sigmoid(self._project(acts) - self._bias)

    def score(self, acts: Tensor, labels: Tensor) -> float:
        preds = self.predict(acts)
        return float((preds == labels.long()).float().mean())

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {"type": "mm", "direction": self.direction, "bias": self._bias, "use_cov": self._use_cov},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MassMeanProbe":
        state = torch.load(path, weights_only=True)
        probe = cls()
        probe.direction = state["direction"]
        probe._bias = float(state["bias"])
        probe._use_cov = bool(state["use_cov"])
        return probe
