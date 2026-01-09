"""Configuration classes for SSNMF variants.

This module defines dataclass-based configuration objects for all SSNMF implementations.
The configuration hierarchy follows the class hierarchy to maintain consistency.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BaseNMFConfig:
    """Base configuration for all NMF variants.

    Attributes:
        n_components: Number of components (rank of factorization)
        max_iter: Maximum number of iterations
        tol: Relative tolerance for convergence
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
        eps: Small value for numerical stability
    """
    n_components: int
    max_iter: int = 100
    tol: float = 1e-4
    random_state: int = 42
    verbose: bool = False
    eps: float = 1e-10


@dataclass
class SSNMFConfig:
    """Configuration for semi-supervised NMF.

    Attributes:
        base_config: Base NMF configuration
        alpha: Label constraint regularization strength (default: 1.0)
               Higher values enforce stronger label constraints.
               Set to 0.0 for unsupervised mode.
    """
    base_config: BaseNMFConfig
    alpha: float = 1.0


@dataclass
class SSNMFFrobeniusConfig:
    """Configuration for Frobenius norm SSNMF.

    This uses the standard Frobenius norm for both reconstruction error
    and label constraints.

    Attributes:
        ssnmf_config: Semi-supervised NMF configuration
    """
    ssnmf_config: SSNMFConfig


@dataclass
class SSNMFKLConfig:
    """Configuration for KL divergence SSNMF.

    This uses KL divergence for reconstruction error and Frobenius norm
    for label constraints (hybrid approach).

    Attributes:
        ssnmf_config: Semi-supervised NMF configuration
    """
    ssnmf_config: SSNMFConfig


@dataclass
class SSNMFDConfig:
    """Configuration for SSNMF with graph regularization.

    This adds Laplacian graph regularization to the base SSNMF formulation.
    Can be used with either Frobenius or KL divergence variants.

    Attributes:
        ssnmf_config: Semi-supervised NMF configuration
        gamma: Graph regularization strength (default: 0.1)
               Higher values enforce smoother solutions along the graph.
        n_neighbors: Number of neighbors for k-NN graph construction (default: 10)
    """
    ssnmf_config: SSNMFConfig
    gamma: float = 0.1
    n_neighbors: int = 10


@dataclass
class SSKNMFConfig:
    """Configuration for Semi-Supervised Kernel Non-negative Matrix Factorization.

    This configuration is for SS-KNMF which uses kernel matrices K ≈ HH^T
    instead of the standard X ≈ WH decomposition.

    Attributes:
        n_clusters: Number of clusters (rank of factorization)
        alpha: Label constraint regularization strength (default: 0.1)
               Higher values enforce stronger label constraints.
               Set to 0.0 for unsupervised mode.
        max_iter: Maximum number of iterations (default: 200)
        tol: Relative tolerance for convergence (default: 1e-4)
        random_state: Random seed for reproducibility (default: 42)
        verbose: Whether to print progress information (default: False)
        eps: Small value for numerical stability (default: 1e-10)
    """
    n_clusters: int
    alpha: float = 0.1
    max_iter: int = 200
    tol: float = 1e-4
    random_state: int = 42
    verbose: bool = False
    eps: float = 1e-10


# Convenience factory functions for creating configurations

def create_frobenius_config(
    n_components: int,
    alpha: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    verbose: bool = False,
    eps: float = 1e-10,
) -> SSNMFFrobeniusConfig:
    """Create a Frobenius norm SSNMF configuration.

    Args:
        n_components: Number of components
        alpha: Label constraint strength
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
        verbose: Verbosity flag
        eps: Numerical stability epsilon

    Returns:
        SSNMFFrobeniusConfig instance
    """
    base_config = BaseNMFConfig(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
        eps=eps,
    )
    ssnmf_config = SSNMFConfig(base_config=base_config, alpha=alpha)
    return SSNMFFrobeniusConfig(ssnmf_config=ssnmf_config)


def create_kl_config(
    n_components: int,
    alpha: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    verbose: bool = False,
    eps: float = 1e-10,
) -> SSNMFKLConfig:
    """Create a KL divergence SSNMF configuration.

    Args:
        n_components: Number of components
        alpha: Label constraint strength
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
        verbose: Verbosity flag
        eps: Numerical stability epsilon

    Returns:
        SSNMFKLConfig instance
    """
    base_config = BaseNMFConfig(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
        eps=eps,
    )
    ssnmf_config = SSNMFConfig(base_config=base_config, alpha=alpha)
    return SSNMFKLConfig(ssnmf_config=ssnmf_config)


def create_ssnmfd_config(
    n_components: int,
    alpha: float = 1.0,
    gamma: float = 0.1,
    n_neighbors: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    verbose: bool = False,
    eps: float = 1e-10,
) -> SSNMFDConfig:
    """Create a graph-regularized SSNMF configuration.

    Args:
        n_components: Number of components
        alpha: Label constraint strength
        gamma: Graph regularization strength
        n_neighbors: Number of neighbors for k-NN graph
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
        verbose: Verbosity flag
        eps: Numerical stability epsilon

    Returns:
        SSNMFDConfig instance
    """
    base_config = BaseNMFConfig(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
        eps=eps,
    )
    ssnmf_config = SSNMFConfig(base_config=base_config, alpha=alpha)
    return SSNMFDConfig(
        ssnmf_config=ssnmf_config,
        gamma=gamma,
        n_neighbors=n_neighbors,
    )
