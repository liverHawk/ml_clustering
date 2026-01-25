"""Semi-Supervised NMF with KL Divergence.

This module implements SSNMF using KL divergence for the reconstruction error
and Frobenius norm for label constraints (hybrid approach).

Mathematical formulation:
    Unlabeled: min D_KL(X||WH) subject to W≥0, H≥0
    Labeled:   min D_KL(X||WH) + α||W_L - W̄_L||²_F subject to W≥0, H≥0

where D_KL(X||WH) = Σᵢⱼ [Xᵢⱼ log(Xᵢⱼ/(WH)ᵢⱼ) - Xᵢⱼ + (WH)ᵢⱼ]
"""

import numpy as np
from typing import Optional

from .ssnmf_refactored import SSNMF
from .configs import SSNMFKLConfig


class SSNMFKL(SSNMF):
    """SSNMF with KL divergence.

    This class implements KL divergence-based SSNMF with the following
    update rules (from CHECK_DOC.md):

    Unlabeled updates:
        H ← H ⊙ [W^T (X/WH)] / [W^T 1 + ε]
        W ← W ⊙ [(X/WH) H^T] / [1 H^T + ε]

    Labeled updates:
        H ← H ⊙ [W^T (X/WH)] / [W^T 1 + ε]
        W ← W ⊙ [(X/WH) H^T + α W̄_L/W] / [1 H^T + α 1_L/W + ε]

    Note: The label constraint uses a hybrid approach combining KL divergence
    for reconstruction and Frobenius norm for label constraints.
    """

    def __init__(self, config: SSNMFKLConfig):
        """Initialize SSNMFKL.

        Args:
            config: SSNMFKLConfig instance
        """
        super().__init__(config.ssnmf_config)
        self.eps = config.ssnmf_config.base_config.eps

    def _compute_reconstruction_error(self, X: np.ndarray) -> float:
        """Compute KL divergence reconstruction error.

        Computes D_KL(X||WH) = Σᵢⱼ [Xᵢⱼ log(Xᵢⱼ/(WH)ᵢⱼ) - Xᵢⱼ + (WH)ᵢⱼ]

        Args:
            X: Data matrix

        Returns:
            KL divergence
        """
        WH = self.W @ self.H + self.eps

        # D_KL(X||WH) = X*log(X/WH) - X + WH
        term1 = X * np.log((X + self.eps) / WH)
        term2 = -X
        term3 = WH

        return np.sum(term1 + term2 + term3)

    def _update_H(self, X: np.ndarray, W: np.ndarray):
        """KL divergence update for H.

        From CHECK_DOC.md Line 145-148:
        H ← H ⊙ [W^T (X/WH)] / [W^T 1 + ε]

        Args:
            X: Data matrix
            W: Factor matrix
        """
        WH = W @ self.H + self.eps

        # Numerator: W^T @ (X / WH)
        numerator = W.T @ (X / WH)

        # Denominator: sum over samples (axis 0)
        # This is equivalent to W^T @ 1 where 1 is a column vector of ones
        denominator = np.sum(W, axis=0, keepdims=True).T + self.eps

        self.H *= numerator / denominator
        self.H = np.maximum(self.H, self.eps)

    def _update_W_unlabeled(self, X: np.ndarray, H: np.ndarray):
        """KL divergence update for W without labels.

        From CHECK_DOC.md Line 153-156:
        W ← W ⊙ [(X/WH) H^T] / [1 H^T + ε]

        Args:
            X: Data matrix
            H: Coefficient matrix
        """
        WH = self.W @ H + self.eps

        # Numerator: (X / WH) @ H^T
        numerator = (X / WH) @ H.T

        # Denominator: sum over features (axis 1)
        # This is equivalent to 1 @ H^T where 1 is a row vector of ones
        denominator = np.sum(H, axis=1, keepdims=True).T + self.eps

        self.W *= numerator / denominator
        self.W = np.maximum(self.W, self.eps)

    def _update_W_labeled(self, X: np.ndarray, H: np.ndarray):
        """KL divergence update for W with label constraints.

        This uses a hybrid approach: KL divergence for reconstruction
        and Frobenius norm for label constraints.

        The update rule is derived from:
        min D_KL(X||WH) + α||W_L - W̄_L||²_F

        Args:
            X: Data matrix
            H: Coefficient matrix
        """
        n_samples = X.shape[0]
        WH = self.W @ H + self.eps

        # KL reconstruction gradient part
        kl_numerator = (X / WH) @ H.T
        kl_denominator = np.sum(H, axis=1, keepdims=True).T

        # Label constraint (Frobenius-style)
        # Create sparse label constraint matrix
        label_constraint = np.zeros_like(self.W)
        label_constraint[self.labeled_indices] = self.W_label

        # Create label mask
        label_mask = np.zeros((n_samples, 1))
        label_mask[self.labeled_indices] = 1

        # Combined update (approximation)
        # For KL + Frobenius constraint, we use an approximation:
        # The Frobenius constraint is added similarly to the Frobenius case
        numerator = kl_numerator + self.alpha * label_constraint / (self.W + self.eps)
        denominator = kl_denominator + self.alpha * label_mask / (self.W + self.eps)

        self.W *= numerator / (denominator + self.eps)
        self.W = np.maximum(self.W, self.eps)


class SSNMFDKL(SSNMFKL):
    """SSNMF with KL divergence and graph regularization.

    This extends SSNMFKL with graph-based regularization using
    the graph Laplacian.

    Objective:
        min D_KL(X||WH) + α||W_L - W̄_L||²_F + γ Tr(H L H^T)

    where:
        L: Graph Laplacian matrix
        γ: Graph regularization strength

    The H update rule becomes:
        H ← H ⊙ [W^T (X/WH) + γ H A] / [W^T 1 + γ H D + ε]

    where A is adjacency matrix and D is degree matrix (L = D - A).
    """

    def __init__(self, config):
        """Initialize SSNMFDKL.

        Args:
            config: SSNMFDConfig instance
        """
        from .configs import SSNMFKLConfig, SSNMFDConfig

        # Ensure correct config type
        if not isinstance(config, SSNMFDConfig):
            raise TypeError("Expected SSNMFDConfig")

        # Create KL config for parent
        kl_config = SSNMFKLConfig(ssnmf_config=config.ssnmf_config)
        super().__init__(kl_config)

        # Graph regularization parameters
        self.gamma = config.gamma
        self.n_neighbors = config.n_neighbors

        # Graph matrices (will be computed in fit)
        self.A: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.L: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        labeled_indices: Optional[np.ndarray] = None,
    ) -> "SSNMFDKL":
        """Fit the model with graph regularization.

        This builds the k-NN graph before running the optimization.

        Args:
            X: Data matrix
            labels: Optional label constraints
            labeled_indices: Optional labeled sample indices

        Returns:
            self: Fitted model
        """
        # Build graph before fitting
        self._prepare_graph(X)
        return super().fit(X, labels, labeled_indices)

    def _prepare_graph(self, X: np.ndarray):
        """Build k-NN graph and compute Laplacian.

        Args:
            X: Data matrix
        """
        from clustering_methods.utils.graph import build_laplacian

        self.A, self.D, self.L = build_laplacian(X, self.n_neighbors)

    def _update_H(self, X: np.ndarray, W: np.ndarray):
        """Update H with KL divergence and graph regularization.

        Combines KL divergence gradient with graph term:
        H ← H ⊙ [W^T (X/WH) + γ H A] / [W^T 1 + γ H D + ε]

        Args:
            X: Data matrix
            W: Factor matrix
        """
        if self.L is None:
            raise RuntimeError("Graph not prepared. Call fit() first.")

        WH = W @ self.H + self.eps

        # KL divergence part
        kl_numerator = W.T @ (X / WH)
        kl_denominator = np.sum(W, axis=0, keepdims=True).T

        # Graph regularization terms
        HA = self.H @ self.A
        HD = self.H @ self.D

        # Combined update
        numerator = kl_numerator + self.gamma * HA
        denominator = kl_denominator + self.gamma * HD + self.eps

        self.H *= numerator / denominator
        self.H = np.maximum(self.H, self.eps)

    def _compute_objective(self, X: np.ndarray) -> float:
        """Compute objective with graph regularization.

        This adds the graph term γ Tr(H L H^T) to the base objective.

        Args:
            X: Data matrix

        Returns:
            Total objective value
        """
        # Base objective (KL divergence + label constraint)
        base_obj = super()._compute_objective(X)

        # Graph regularization term
        if self.L is not None:
            # Tr(H L H^T) = Tr(H^T H L)
            graph_term = np.trace(self.H @ self.L @ self.H.T)
            return base_obj + self.gamma * graph_term

        return base_obj
