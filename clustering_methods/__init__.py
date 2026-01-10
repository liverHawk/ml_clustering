"""Clustering methods module.

This module provides various clustering and matrix factorization methods,
including Semi-Supervised Non-negative Matrix Factorization (SSNMF).
"""

# New API - Refactored implementations
from .ssnmf_refactored import (
    SSNMF,
    SSNMFFrobenius,
    SSNMFDFrobenius,
)
from .ssnmf_kl_refactored import (
    SSNMFKL,
    SSNMFDKL,
)
from .configs import (
    BaseNMFConfig,
    SSNMFConfig,
    SSNMFFrobeniusConfig,
    SSNMFKLConfig,
    SSNMFDConfig,
    SSKNMFConfig,
    create_frobenius_config,
    create_kl_config,
    create_ssnmfd_config,
)
from .gower_ssknmf import (
    GowerSSKNMF,
    SSKNMF,
    gower_distance_vectorized,
    gower_to_kernel,
)

# Backward compatibility - Old implementations
# These are kept for compatibility with existing code like sample_ssnmf_d.py
from .ssnmf import (
    BaseNMF,
    SSNMF as SSNMF_OLD,
    SSNMFD as SSNMFD_OLD,
    BaseNMFConfig as BaseNMFConfig_OLD,
    SSNMFDConfig as SSNMFDConfig_OLD,
)

# For backward compatibility, export old names under 'ssnmf' namespace
# This allows code like: from clustering_methods import ssnmf; ssnmf.SSNMFD(...)
class _BackwardCompatNamespace:
    """Namespace for backward compatibility."""
    SSNMF = SSNMF_OLD
    SSNMFD = SSNMFD_OLD
    BaseNMFConfig = BaseNMFConfig_OLD
    SSNMFDConfig = SSNMFDConfig_OLD

ssnmf = _BackwardCompatNamespace()

__all__ = [
    # New API
    "SSNMF",
    "SSNMFFrobenius",
    "SSNMFDFrobenius",
    "SSNMFKL",
    "SSNMFDKL",
    "BaseNMFConfig",
    "SSNMFConfig",
    "SSNMFFrobeniusConfig",
    "SSNMFKLConfig",
    "SSNMFDConfig",
    "SSKNMFConfig",
    "create_frobenius_config",
    "create_kl_config",
    "create_ssnmfd_config",
    # Gower SS-KNMF
    "GowerSSKNMF",
    "SSKNMF",
    "gower_distance_vectorized",
    "gower_to_kernel",
    # Backward compatibility
    "BaseNMF",
    "SSNMF_OLD",
    "SSNMFD_OLD",
    "ssnmf",
]
