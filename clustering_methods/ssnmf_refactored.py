"""Refactored Semi-Supervised Non-negative Matrix Factorization (SSNMF).

This module implements the semi-supervised NMF with proper label constraints
as defined in CHECK_DOC.md. It supports both labeled and unlabeled modes with
Frobenius norm.

Mathematical formulation:
    Unlabeled: min ||X - WH||²_F subject to W≥0, H≥0
    Labeled:   min ||X - WH||²_F + α||W_L - W̄_L||²_F subject to W≥0, H≥0

where W_L are rows of W corresponding to labeled samples, and W̄_L is the
label constraint matrix.
"""

import abc
import numpy as np
import logging
from typing import Optional

from .ssnmf import BaseNMF, BaseNMFConfig
from .configs import SSNMFConfig, SSNMFFrobeniusConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSNMF(BaseNMF):
    """Semi-Supervised Non-negative Matrix Factorization base class.

    This abstract base class implements the label constraint logic and
    common optimization framework. Subclasses must implement divergence-specific
    update rules.

    The class supports both labeled and unlabeled modes:
    - Unlabeled: fit(X) - Standard NMF
    - Labeled: fit(X, labels, labeled_indices) - SSNMF with label constraints

    Attributes:
        config: Base NMF configuration
        alpha: Label constraint regularization strength
        W: Factor matrix (n_samples × n_components)
        H: Coefficient matrix (n_components × n_features)
        labeled_mode: Whether labels are provided
        W_label: Label constraint matrix (n_labeled × n_components)
        labeled_indices: Indices of labeled samples
        n_labeled: Number of labeled samples
    """

    def __init__(self, config: SSNMFConfig):
        """Initialize SSNMF.

        Args:
            config: SSNMFConfig containing base config and alpha parameter
        """
        super().__init__(config.base_config)
        self.alpha = config.alpha

        # Label-related attributes
        self.labeled_mode: bool = False
        self.W_label: Optional[np.ndarray] = None
        self.labeled_indices: Optional[np.ndarray] = None
        self.n_labeled: int = 0

    def fit(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        labeled_indices: Optional[np.ndarray] = None,
    ) -> "SSNMF":
        """Fit the SSNMF model.

        Parameters:
            X: Data matrix, shape (n_samples, n_features)
               Must be non-negative
            labels: Label constraint matrix W̄_L, shape (n_labeled, n_components)
                   Optional, for semi-supervised mode
            labeled_indices: Indices of labeled samples, shape (n_labeled,)
                            Optional, must be provided with labels

        Returns:
            self: Fitted model

        Raises:
            ValueError: If input validation fails
        """
        X = self._check_input(X)
        n_samples, n_features = X.shape

        # Setup labels if provided
        if labels is not None and labeled_indices is not None:
            self._setup_labels(labels, labeled_indices, n_samples)
            self.labeled_mode = True
        else:
            self.labeled_mode = False

        # Initialize W and H if not already initialized
        if self.W is None or self.H is None:
            rng = np.random.default_rng(self.config.random_state)
            self.W = rng.random(size=(n_samples, self.config.n_components)) + 1e-4
            self.H = rng.random(size=(self.config.n_components, n_features)) + 1e-4

        # Run optimization loop
        return self._fit_loop(X)

    def _setup_labels(
        self, labels: np.ndarray, labeled_indices: np.ndarray, n_samples: int
    ):
        """Validate and store label information.

        Args:
            labels: Label constraint matrix
            labeled_indices: Indices of labeled samples
            n_samples: Total number of samples in X

        Raises:
            ValueError: If validation fails
        """
        self.labeled_indices = np.asarray(labeled_indices, dtype=int)
        self.W_label = np.asarray(labels, dtype=float)
        self.n_labeled = len(self.labeled_indices)

        # Validation
        if self.n_labeled != self.W_label.shape[0]:
            raise ValueError(
                f"Mismatch: labeled_indices has {self.n_labeled} elements "
                f"but labels has {self.W_label.shape[0]} rows"
            )

        if self.W_label.shape[1] != self.config.n_components:
            raise ValueError(
                f"Labels must have n_components={self.config.n_components} columns, "
                f"got {self.W_label.shape[1]}"
            )

        if np.any(self.labeled_indices >= n_samples) or np.any(
            self.labeled_indices < 0
        ):
            raise ValueError(
                f"labeled_indices out of bounds [0, {n_samples}): "
                f"min={self.labeled_indices.min()}, max={self.labeled_indices.max()}"
            )

        if np.any(self.W_label < 0):
            raise ValueError("Labels must be non-negative")

    def _fit_loop(self, X: np.ndarray) -> "SSNMF":
        """Main optimization loop with convergence checking.

        Args:
            X: Data matrix

        Returns:
            self: Fitted model
        """
        prev_err = None

        for iteration in range(1, self.config.max_iter + 1):
            # Update W and H using BaseNMF interface
            self._update_W(X, self.H)
            self._update_H(X, self.W)

            # Compute objective
            err = self._compute_objective(X)

            # Check convergence
            if self._check_convergence(err, prev_err, iteration):
                if self.config.verbose:
                    logger.info("Converged.")
                break

            prev_err = err

        return self

    def _update_W(self, X: np.ndarray, H: np.ndarray):
        """Update W (dispatches to labeled/unlabeled version).

        This method implements the BaseNMF abstract method and dispatches
        to the appropriate labeled or unlabeled update method.

        Args:
            X: Data matrix
            H: Coefficient matrix
        """
        if self.labeled_mode:
            self._update_W_labeled(X, H)
        else:
            self._update_W_unlabeled(X, H)

    def _check_convergence(
        self, err: float, prev_err: Optional[float], iteration: int
    ) -> bool:
        """Check if optimization has converged.

        Args:
            err: Current objective value
            prev_err: Previous objective value
            iteration: Current iteration number

        Returns:
            True if converged, False otherwise
        """
        if prev_err is None or not np.isfinite(prev_err):
            rel = np.inf
        else:
            rel = abs(prev_err - err) / (abs(prev_err) + self.config.eps)

        if self.config.verbose:
            logger.info(f"Iter {iteration:03d}: err={err:.6f}, rel={rel:.2e}")

        return rel < self.config.tol

    def _compute_objective(self, X: np.ndarray) -> float:
        """Compute full objective function.

        This includes reconstruction error and (if in labeled mode) label constraint.

        Args:
            X: Data matrix

        Returns:
            Total objective value
        """
        recon_err = self._compute_reconstruction_error(X)

        if self.labeled_mode:
            W_labeled = self.W[self.labeled_indices]
            label_err = np.sum((W_labeled - self.W_label) ** 2)
            return recon_err + self.alpha * label_err

        return recon_err

    @abc.abstractmethod
    def _compute_reconstruction_error(self, X: np.ndarray) -> float:
        """Compute divergence-specific reconstruction error.

        Args:
            X: Data matrix

        Returns:
            Reconstruction error
        """
        pass

    @abc.abstractmethod
    def _update_H(self, X: np.ndarray, W: np.ndarray):
        """Update H (may differ by divergence).

        Args:
            X: Data matrix
            W: Factor matrix
        """
        pass

    @abc.abstractmethod
    def _update_W_unlabeled(self, X: np.ndarray, H: np.ndarray):
        """Update W without label constraints.

        Args:
            X: Data matrix
            H: Coefficient matrix
        """
        pass

    @abc.abstractmethod
    def _update_W_labeled(self, X: np.ndarray, H: np.ndarray):
        """Update W with label constraints.

        Args:
            X: Data matrix
            H: Coefficient matrix
        """
        pass


class SSNMFFrobenius(SSNMF):
    """SSNMF with Frobenius norm.

    This class implements the Frobenius norm-based SSNMF with the following
    update rules (from CHECK_DOC.md):

    Unlabeled updates:
        H ← H ⊙ (W^T X) / (W^T W H + ε)
        W ← W ⊙ (X H^T) / (W H H^T + ε)

    Labeled updates:
        H ← H ⊙ (W^T X) / (W^T W H + ε)
        W ← W ⊙ (X H^T + α W̄_L δ_L) / (W H H^T + α δ_L + ε)

    where δ_L is an indicator matrix (1 for labeled samples, 0 elsewhere).
    """

    def __init__(self, config: SSNMFFrobeniusConfig):
        """Initialize SSNMFFrobenius.

        Args:
            config: SSNMFFrobeniusConfig instance
        """
        super().__init__(config.ssnmf_config)
        self.eps = config.ssnmf_config.base_config.eps

    def _compute_reconstruction_error(self, X: np.ndarray) -> float:
        """Compute Frobenius norm reconstruction error.

        Computes ||X - WH||²_F = Σᵢⱼ (Xᵢⱼ - (WH)ᵢⱼ)²

        Args:
            X: Data matrix

        Returns:
            Frobenius norm squared
        """
        reconstruction = self.W @ self.H
        diff = X - reconstruction
        return np.sum(diff**2)

    def _update_H(self, X: np.ndarray, W: np.ndarray):
        """Standard Frobenius update for H.

        H ← H ⊙ (W^T X) / (W^T W H + ε)

        Args:
            X: Data matrix
            W: Factor matrix
        """
        numerator = W.T @ X
        denominator = (W.T @ W) @ self.H + self.eps

        self.H *= numerator / denominator
        self.H = np.maximum(self.H, self.eps)

    def _update_W_unlabeled(self, X: np.ndarray, H: np.ndarray):
        """Standard Frobenius update for W.

        W ← W ⊙ (X H^T) / (W H H^T + ε)

        Args:
            X: Data matrix
            H: Coefficient matrix
        """
        numerator = X @ H.T
        denominator = self.W @ (H @ H.T) + self.eps

        self.W *= numerator / denominator
        self.W = np.maximum(self.W, self.eps)

    def _update_W_labeled(self, X: np.ndarray, H: np.ndarray):
        """Frobenius update with label constraints.

        From CHECK_DOC.md line 258:
        W ← W ⊙ (X H^T + α W̄_L δ_L) / (W H H^T + α δ_L + ε)

        Args:
            X: Data matrix
            H: Coefficient matrix
        """
        n_samples = X.shape[0]

        # Create label constraint matrix (sparse representation)
        label_constraint = np.zeros_like(self.W)
        label_constraint[self.labeled_indices] = self.W_label

        # Create label mask (indicator for labeled samples)
        label_mask = np.zeros((n_samples, 1))
        label_mask[self.labeled_indices] = 1

        # Multiplicative update with label constraint
        numerator = X @ H.T + self.alpha * label_constraint
        denominator = self.W @ (H @ H.T) + self.alpha * label_mask + self.eps

        self.W *= numerator / denominator
        self.W = np.maximum(self.W, self.eps)


class SSNMFDFrobenius(SSNMFFrobenius):
    """SSNMF with Frobenius norm and graph regularization.

    This extends SSNMFFrobenius with graph-based regularization using
    the graph Laplacian.

    Objective:
        min ||X - WH||²_F + α||W_L - W̄_L||²_F + γ Tr(H L H^T)

    where:
        L: Graph Laplacian matrix
        γ: Graph regularization strength

    The H update rule becomes:
        H ← H ⊙ [W^T X + γ H A] / [W^T W H + γ H D + ε]

    where A is adjacency matrix and D is degree matrix (L = D - A).
    """

    def __init__(self, config):
        """Initialize SSNMFDFrobenius.

        Args:
            config: SSNMFDConfig instance
        """
        from .configs import SSNMFFrobeniusConfig, SSNMFDConfig

        # Ensure correct config type
        if not isinstance(config, SSNMFDConfig):
            raise TypeError("Expected SSNMFDConfig")

        # Create Frobenius config for parent
        frob_config = SSNMFFrobeniusConfig(ssnmf_config=config.ssnmf_config)
        super().__init__(frob_config)

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
    ) -> "SSNMFDFrobenius":
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
        from .utils.graph import build_laplacian

        self.A, self.D, self.L = build_laplacian(X, self.n_neighbors)

    def _update_H(self, X: np.ndarray, W: np.ndarray):
        """Update H with graph regularization.

        From current SSNMFD implementation (ssnmf.py lines 122-141):
        H ← H ⊙ [W^T X + γ H A] / [W^T W H + γ H D + ε]

        Args:
            X: Data matrix
            W: Factor matrix
        """
        if self.L is None:
            raise RuntimeError("Graph not prepared. Call fit() first.")

        WT_X = W.T @ X
        WT_W_H = (W.T @ W) @ self.H

        # Graph regularization terms
        HA = self.H @ self.A
        HD = self.H @ self.D

        numerator = WT_X + self.gamma * HA
        denominator = WT_W_H + self.gamma * HD + self.eps

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
        # Base objective (reconstruction + label constraint)
        base_obj = super()._compute_objective(X)

        # Graph regularization term
        if self.L is not None:
            # Tr(H L H^T) = Tr(H^T H L)
            graph_term = np.trace(self.H @ self.L @ self.H.T)
            return base_obj + self.gamma * graph_term

        return base_obj
