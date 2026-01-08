from typing import Optional
import abc
import numpy as np
import logging
from dataclasses import dataclass

from .utils.graph import build_laplacian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaseNMFConfig:
    n_components: int
    max_iter: int
    tol: float
    random_state: int
    verbose: bool


class BaseNMF(abc.ABC):
    def __init__(self, config: BaseNMFConfig):
        self.config = config

        self.W = None
        self.H = None

    @abc.abstractmethod
    def _update_W(self, X, H):
        """W update method"""
        pass

    @abc.abstractmethod
    def _update_H(self, X, W):
        """H update method"""
        pass

    def _check_input(self, X):
        X = np.asarray(X, dtype=float)
        if np.any(X < 0):
            raise ValueError("X must be non-negative.")
        return X

    def fit(self, X):
        X = self._check_input(X)
        m, n = X.shape
        rng = np.random.default_rng(self.config.random_state)

        if self.W is None or self.H is None:
            self.W = rng.random(size=(m, self.config.n_components)) + 1e-4
            self.H = rng.random(size=(self.config.n_components, n)) + 1e-4

        prev_err = None
        for it in range(1, self.config.max_iter + 1):
            self._update_W(X, self.H)
            self._update_H(X, self.W)

            err = np.linalg.norm(X - self.W @ self.H, "fro") ** 2
            if prev_err is None or not np.isfinite(prev_err):
                rel = np.inf
            else:
                rel = abs(prev_err - err) / (prev_err + 1e-10)

            if self.config.verbose:
                logger.info(f"Iter {it:03d}: err = {err:.6f}, rel = {rel:.2e}")
            if rel < self.config.tol:
                if self.config.verbose:
                    logger.info("Converged.")
                break
            prev_err = err
        return self

    def reconstruct(self):
        if self.W is None or self.H is None:
            raise ValueError("Model is not fitted yet.")
        return self.W @ self.H


class SSNMF(BaseNMF):
    def __init__(self, config: BaseNMFConfig, lambda_reg: float = 0.1):
        import warnings
        warnings.warn(
            "SSNMF from ssnmf.py is deprecated and does not implement proper "
            "semi-supervised label constraints. Use SSNMFFrobenius from "
            "clustring_methods.ssnmf_refactored instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(config)
        self.lambda_reg = lambda_reg
        # self.config = config

    def _update_W(self, X, H):
        numer = X @ H.T
        denom = self.W @ (H @ H.T)
        denom = np.where(denom == 0, 1e-10, denom)
        self.W *= numer / denom
        self.W = np.maximum(self.W, 1e-10)

    def _update_H(self, X, W):
        numer = W.T @ X
        denom = (W.T @ W) @ self.H + self.lambda_reg
        denom = np.where(denom == 0, 1e-10, denom)
        self.H *= numer / denom
        self.H = np.maximum(self.H, 1e-10)


@dataclass
class SSNMFDConfig:
    ssnmf_config: BaseNMFConfig
    lambda_reg: float = 0.1
    gamma_reg: float = 0.1
    n_neighbors: int = 10


class SSNMFD(SSNMF):
    def __init__(self, config: SSNMFDConfig):
        import warnings
        warnings.warn(
            "SSNMFD from ssnmf.py is deprecated. Use SSNMFDFrobenius from "
            "clustring_methods.ssnmf_refactored instead for proper label "
            "constraint support.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(config.ssnmf_config, config.lambda_reg)
        self.gamma_reg = config.gamma_reg
        self.n_neighbors = config.n_neighbors

        self.A: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.L: Optional[np.ndarray] = None

    def prepare_graph(self, X):
        self.A, self.D, self.L = build_laplacian(X, self.n_neighbors)

    def _update_H(self, X, W):
        if self.L is None:
            self.prepare_graph(X)

        assert self.H is not None, "H must be initialized before update"
        assert self.W is not None, "self.W must be initialized"
        assert self.A is not None and self.D is not None, "Graph matrices must be prepared"

        WT_X = W.T @ X
        WT_W = W.T @ self.W

        HA = self.H @ self.A
        HD = self.H @ self.D

        numer = WT_X + self.gamma_reg * HA
        denom = WT_W @ self.H + self.lambda_reg + self.gamma_reg * HD
        denom = np.where(denom == 0, 1e-10, denom)

        self.H *= numer / denom
        self.H = np.maximum(self.H, 1e-10)
