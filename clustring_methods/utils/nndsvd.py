from numpy.typing import NDArray
from typing import TypeAlias, Tuple
import numpy as np

from scipy.linalg import svd
from dataclasses import dataclass


SVDResult: TypeAlias = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]

@dataclass
class NNDSVDConfig:
    X: np.ndarray
    rank: int
    variant: str = "zero"
    eps: float = 1e-6


def nndsvd_initialization(config: NNDSVDConfig):
    if np.any(config.X < 0):
        raise ValueError("X must be non-negative.")

    # 入力を安全にfloat配列へ変換（shape解釈の誤用を防ぐ）
    X = np.array(config.X, dtype=float, copy=True)

    U, s, VT = svd(X, full_matrices=False)
    svd_rank = s.size

    r = min(svd_rank, config.rank)
    if r <= 0:
        raise ValueError("Rank must be positive.")

    U = U[:, :r]
    s = s[:r]
    V = VT[:r, :].T

    m, n = config.X.shape
    W = np.zeros((m, r))
    H = np.zeros((r, n))

    sqrt_s0 = np.sqrt(s[0])
    W[:, 0] = sqrt_s0 * np.abs(U[:, 0])
    # Vは形状(n, r)なので列ベクトルを使用する
    H[0, :] = sqrt_s0 * np.abs(V[:, 0])

    for k in range(1, r):
        uk = U[:, k]
        vk = V[:, k]

        uk_pos = np.maximum(uk, 0.0)
        uk_neg = np.minimum(uk, 0.0)
        vk_pos = np.maximum(vk, 0.0)
        vk_neg = np.minimum(vk, 0.0)

        norm_uk_pos = np.linalg.norm(uk_pos)
        norm_uk_neg = np.linalg.norm(uk_neg)
        norm_vk_pos = np.linalg.norm(vk_pos)
        norm_vk_neg = np.linalg.norm(vk_neg)

        norm_pos = norm_uk_pos * norm_vk_pos
        norm_neg = norm_uk_neg * norm_vk_neg

        if norm_pos < norm_neg:
            wk = np.sqrt(s[k]) * uk_neg / (norm_uk_neg + 1e-12)
            hk = np.sqrt(s[k]) * vk_neg / (norm_vk_neg + 1e-12)
        else:
            wk = np.sqrt(s[k]) * uk_pos / (norm_uk_pos + 1e-12)
            hk = np.sqrt(s[k]) * vk_pos / (norm_uk_pos + 1e-12)

        W[:, k] = wk
        H[k, :] = hk

    if config.variant == "average":
        avg = config.X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif config.variant == "random":
        rng = np.random.default_rng(42)
        W[W == 0] = rng.uniform(0, config.eps, size=W[W == 0].shape)
        H[H == 0] = rng.uniform(0, config.eps, size=H[H == 0].shape)

    return W, H
