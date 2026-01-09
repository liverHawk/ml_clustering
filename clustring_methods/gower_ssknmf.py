"""Gower距離ベースSemi-Supervised Kernel Non-negative Matrix Factorization (SS-KNMF).

This module implements SS-KNMF using Gower distance for mixed-type data
(categorical and numerical variables) without requiring one-hot encoding.

Mathematical formulation:
    min ||K - HH^T||²_F + α||H_labeled - H_constraint||²_F
    subject to H ≥ 0

where:
    K: Kernel matrix (n×n) derived from Gower distance
    H: Membership matrix (k×n) where k is the number of clusters
    H_constraint: Label constraint matrix for labeled samples
    α: Regularization strength for label constraints
"""

import numpy as np
import polars as pl
import logging
from typing import Optional, List, Union
from dataclasses import dataclass

from .configs import SSKNMFConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gower_distance_vectorized(
    df: pl.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> np.ndarray:
    """ベクトル化されたGower距離計算
    
    Gower距離は混合型データ（カテゴリ変数と数値変数）に対して
    統一的に距離を計算する手法です。
    
    Args:
        df: Polars DataFrame
        categorical_cols: カテゴリ変数のカラム名リスト
        numerical_cols: 数値変数のカラム名リスト
    
    Returns:
        Gower距離行列 (n×n)
    
    Note:
        計算量: O(n² * f)
        メモリ: O(n²)
    """
    n = len(df)
    n_features = len(categorical_cols) + len(numerical_cols)
    
    if n_features == 0:
        raise ValueError("At least one feature column must be specified")
    
    # 数値変数の距離計算
    num_dist = np.zeros((n, n))
    if len(numerical_cols) > 0:
        num_data = df.select(numerical_cols).to_numpy()
        ranges = np.ptp(num_data, axis=0)  # max - min
        ranges[ranges == 0] = 1  # ゼロ除算回避
        
        # ブロードキャスト: (n,1,f) - (1,n,f) → (n,n,f)
        for k in range(len(numerical_cols)):
            num_dist += np.abs(
                num_data[:, None, k] - num_data[None, :, k]
            ) / ranges[k]
    
    # カテゴリ変数の距離計算
    cat_dist = np.zeros((n, n))
    if len(categorical_cols) > 0:
        cat_data = df.select(categorical_cols).to_numpy()
        
        # ブロードキャスト: (n,1,f) != (1,n,f) → (n,n,f)
        for k in range(len(categorical_cols)):
            cat_dist += (cat_data[:, None, k] != cat_data[None, :, k]).astype(float)
    
    # Gower距離 = (数値距離 + カテゴリ距離) / 特徴量数
    return (num_dist + cat_dist) / n_features


def gower_to_kernel(
    distance_matrix: np.ndarray,
    method: str = 'linear',
    sigma: Optional[float] = None,
) -> np.ndarray:
    """距離行列をカーネル行列に変換
    
    Args:
        distance_matrix: Gower距離行列 (n×n)
        method: カーネル方法 ('linear', 'rbf', 'exponential')
        sigma: カーネル幅（RBF/Exponential用）。Noneの場合は自動設定
    
    Returns:
        カーネル行列 (n×n)
    """
    if method == 'linear':
        return 1 - distance_matrix
    
    elif method == 'rbf':
        if sigma is None:
            # ヒューリスティック: 距離の中央値を使用
            non_zero_dist = distance_matrix[distance_matrix > 0]
            if len(non_zero_dist) > 0:
                sigma = np.median(non_zero_dist)
            else:
                sigma = 0.5
        
        return np.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))
    
    elif method == 'exponential':
        if sigma is None:
            # ヒューリスティック: 距離の中央値を使用
            non_zero_dist = distance_matrix[distance_matrix > 0]
            if len(non_zero_dist) > 0:
                sigma = np.median(non_zero_dist)
            else:
                sigma = 0.5
        
        return np.exp(-distance_matrix / sigma)
    
    else:
        raise ValueError(f"Unknown kernel method: {method}. "
                        f"Must be one of 'linear', 'rbf', 'exponential'")


class SSKNMF:
    """Semi-Supervised Kernel Non-negative Matrix Factorization.
    
    This class implements SS-KNMF which factorizes a kernel matrix K ≈ HH^T
    with optional label constraints for semi-supervised learning.
    
    Attributes:
        config: SSKNMFConfig instance
        H_: Membership matrix (k×n)
        K_: Kernel matrix (n×n)
        reconstruction_errors_: List of reconstruction errors per iteration
    """
    
    def __init__(self, config: SSKNMFConfig):
        """Initialize SSKNMF.
        
        Args:
            config: SSKNMFConfig instance
        """
        self.config = config
        self.H_: Optional[np.ndarray] = None
        self.K_: Optional[np.ndarray] = None
        self.reconstruction_errors_: List[float] = []
    
    def fit(
        self,
        K: np.ndarray,
        labeled_indices: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> "SSKNMF":
        """Fit the SS-KNMF model.
        
        Args:
            K: Kernel matrix (n×n), must be non-negative
            labeled_indices: Indices of labeled samples (n_labeled,)
            labels: Cluster labels for labeled samples (n_labeled,)
                    Values should be in [0, n_clusters-1]
        
        Returns:
            self: Fitted model
        """
        K = np.asarray(K, dtype=float)
        n = K.shape[0]
        
        if K.shape[1] != n:
            raise ValueError(f"Kernel matrix must be square, got shape {K.shape}")
        
        if np.any(K < 0):
            raise ValueError("Kernel matrix must be non-negative")
        
        self.K_ = K
        
        # ラベル制約の設定
        H_constraint = None
        if labeled_indices is not None and labels is not None:
            H_constraint = self._create_constraint_matrix(
                labeled_indices, labels, n
            )
        
        # Hの初期化
        rng = np.random.default_rng(self.config.random_state)
        self.H_ = rng.random(size=(self.config.n_clusters, n)) + self.config.eps
        
        # 最適化ループ
        self._optimize(K, H_constraint, labeled_indices)
        
        return self
    
    def _create_constraint_matrix(
        self,
        labeled_indices: np.ndarray,
        labels: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """ラベル制約行列を作成
        
        Args:
            labeled_indices: ラベル付きサンプルのインデックス
            labels: クラスタラベル（0からn_clusters-1）
            n_samples: 全サンプル数
        
        Returns:
            制約行列 (k×n)
        """
        labeled_indices = np.asarray(labeled_indices, dtype=int)
        labels = np.asarray(labels, dtype=int)
        
        H_constraint = np.zeros((self.config.n_clusters, n_samples))
        
        for idx, label in zip(labeled_indices, labels):
            if label < 0 or label >= self.config.n_clusters:
                raise ValueError(
                    f"Label {label} out of range [0, {self.config.n_clusters-1}]"
                )
            H_constraint[label, idx] = 1.0
        
        return H_constraint
    
    def _optimize(
        self,
        K: np.ndarray,
        H_constraint: Optional[np.ndarray],
        labeled_indices: Optional[np.ndarray],
    ):
        """最適化ループ
        
        Args:
            K: カーネル行列
            H_constraint: 制約行列
            labeled_indices: ラベル付きサンプルのインデックス
        """
        n = K.shape[0]
        unlabeled_mask = np.ones(n, dtype=bool)
        
        if labeled_indices is not None:
            unlabeled_mask[labeled_indices] = False
        
        prev_err = None
        
        for iteration in range(1, self.config.max_iter + 1):
            H_old = self.H_.copy()
            
            # Hの更新
            self._update_H(K, H_constraint, unlabeled_mask, labeled_indices)
            
            # 再構成誤差の計算
            err = self._compute_reconstruction_error(K, H_constraint, labeled_indices)
            self.reconstruction_errors_.append(err)
            
            # 収束判定
            if prev_err is not None:
                rel = abs(prev_err - err) / (abs(prev_err) + self.config.eps)
                
                if self.config.verbose:
                    logger.info(
                        f"Iteration {iteration:03d}: "
                        f"Reconstruction Error = {err:.6f}, "
                        f"Change = {rel:.2e}"
                    )
                
                if rel < self.config.tol:
                    if self.config.verbose:
                        logger.info("Converged.")
                    break
            else:
                if self.config.verbose:
                    logger.info(
                        f"Iteration {iteration:03d}: "
                        f"Reconstruction Error = {err:.6f}"
                    )
            
            prev_err = err
        
        if iteration == self.config.max_iter and self.config.verbose:
            logger.info("最大反復回数に達しました")
    
    def _update_H(
        self,
        K: np.ndarray,
        H_constraint: Optional[np.ndarray],
        unlabeled_mask: np.ndarray,
        labeled_indices: Optional[np.ndarray],
    ):
        """H行列の更新
        
        更新則:
        H ← H ⊙ √[(KH^T + αH_constraint)^T / (HH^TKH^T + αH)^T]
        
        Args:
            K: カーネル行列
            H_constraint: 制約行列
            unlabeled_mask: ラベルなしサンプルのマスク
            labeled_indices: ラベル付きサンプルのインデックス
        """
        assert self.H_ is not None, "H must be initialized"
        
        # 中間計算
        KH_T = K @ self.H_.T  # (n×k)
        HH_T = self.H_ @ self.H_.T  # (k×k)
        HH_TKH_T = HH_T @ KH_T.T  # (k×n)
        
        # 分子: (KH^T)^T + α * H_constraint
        numerator = KH_T.T  # (k×n)
        if H_constraint is not None and self.config.alpha > 0:
            numerator += self.config.alpha * H_constraint
        
        # 分母: HH^TKH^T + α * H
        denominator = HH_TKH_T  # (k×n)
        if self.config.alpha > 0:
            denominator += self.config.alpha * self.H_
        
        denominator = np.maximum(denominator, self.config.eps)
        
        # ラベルなしサンプルのみ更新
        self.H_[:, unlabeled_mask] = (
            self.H_[:, unlabeled_mask] *
            np.sqrt(numerator[:, unlabeled_mask] / denominator[:, unlabeled_mask])
        )
        
        # ラベル付きサンプルは制約で固定
        if labeled_indices is not None and H_constraint is not None:
            self.H_[:, labeled_indices] = H_constraint[:, labeled_indices]
        
        # 非負性を保証
        self.H_ = np.maximum(self.H_, self.config.eps)
    
    def _compute_reconstruction_error(
        self,
        K: np.ndarray,
        H_constraint: Optional[np.ndarray],
        labeled_indices: Optional[np.ndarray],
    ) -> float:
        """再構成誤差の計算
        
        Args:
            K: カーネル行列
            H_constraint: 制約行列
            labeled_indices: ラベル付きサンプルのインデックス
        
        Returns:
            再構成誤差
        """
        assert self.H_ is not None, "H must be initialized"
        
        # 再構成誤差: ||K - HH^T||²_F
        recon = self.H_.T @ self.H_  # (n×n)
        recon_error = np.linalg.norm(K - recon, 'fro') ** 2
        
        # ラベル制約項: α||H_labeled - H_constraint||²_F
        if H_constraint is not None and labeled_indices is not None:
            H_labeled = self.H_[:, labeled_indices]
            H_constraint_labeled = H_constraint[:, labeled_indices]
            constraint_error = (
                self.config.alpha *
                np.linalg.norm(H_labeled - H_constraint_labeled, 'fro') ** 2
            )
            return recon_error + constraint_error
        
        return recon_error
    
    def predict(self) -> np.ndarray:
        """クラスタラベルを予測
        
        Returns:
            クラスタラベル (n,)
        """
        if self.H_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # 最大所属度のクラスタを選択
        return np.argmax(self.H_, axis=0)
    
    def get_cluster_membership(self) -> np.ndarray:
        """クラスタ所属確率を取得
        
        Returns:
            所属度行列 (k×n)
        """
        if self.H_ is None:
            raise ValueError("Model must be fitted before getting membership")
        
        # 正規化（各行の和が1になるように）
        H_normalized = self.H_.copy()
        row_sums = H_normalized.sum(axis=0, keepdims=True)
        row_sums[row_sums == 0] = 1  # ゼロ除算回避
        H_normalized = H_normalized / row_sums
        
        return H_normalized
    
    def get_reconstruction_errors(self) -> List[float]:
        """再構成誤差の履歴を取得
        
        Returns:
            各反復での再構成誤差のリスト
        """
        return self.reconstruction_errors_


class GowerSSKNMF:
    """Gower距離ベースのSS-KNMF高レベルAPI.
    
    This class provides a high-level interface for SS-KNMF using Gower distance
    for mixed-type data. It handles data preprocessing, distance computation,
    kernel transformation, and clustering in a single interface.
    
    Attributes:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        kernel_method: Kernel transformation method
        kernel_sigma: Kernel width parameter
        model: SSKNMF instance
        distance_matrix_: Computed Gower distance matrix
    """
    
    def __init__(
        self,
        categorical_cols: List[str],
        numerical_cols: List[str],
        n_clusters: int,
        kernel_method: str = 'rbf',
        kernel_sigma: Optional[float] = None,
        alpha: float = 0.1,
        max_iter: int = 200,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: int = 42,
    ):
        """Initialize GowerSSKNMF.
        
        Args:
            categorical_cols: カテゴリ変数のカラム名リスト
            numerical_cols: 数値変数のカラム名リスト
            n_clusters: クラスタ数
            kernel_method: カーネル方法 ('linear', 'rbf', 'exponential')
            kernel_sigma: カーネル幅（Noneの場合は自動設定）
            alpha: ラベル制約の強さ
            max_iter: 最大反復回数
            tol: 収束判定の閾値
            verbose: 進捗表示
            random_state: 乱数シード
        """
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.kernel_method = kernel_method
        self.kernel_sigma = kernel_sigma
        self.distance_matrix_: Optional[np.ndarray] = None
        
        config = SSKNMFConfig(
            n_clusters=n_clusters,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        self.model = SSKNMF(config)
    
    def fit_predict(
        self,
        df: pl.DataFrame,
        labeled_indices: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """学習と予測を実行
        
        Args:
            df: Polars DataFrame
            labeled_indices: ラベル付きサンプルのインデックス
            labels: クラスタラベル
        
        Returns:
            クラスタラベル (n,)
        """
        # Gower距離の計算
        self.distance_matrix_ = gower_distance_vectorized(
            df, self.categorical_cols, self.numerical_cols
        )
        
        # カーネル変換
        K = gower_to_kernel(
            self.distance_matrix_,
            method=self.kernel_method,
            sigma=self.kernel_sigma,
        )
        
        # SS-KNMFの学習
        self.model.fit(K, labeled_indices, labels)
        
        # 予測
        return self.model.predict()
    
    def fit(
        self,
        df: pl.DataFrame,
        labeled_indices: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> "GowerSSKNMF":
        """学習のみ実行
        
        Args:
            df: Polars DataFrame
            labeled_indices: ラベル付きサンプルのインデックス
            labels: クラスタラベル
        
        Returns:
            self: Fitted model
        """
        # Gower距離の計算
        self.distance_matrix_ = gower_distance_vectorized(
            df, self.categorical_cols, self.numerical_cols
        )
        
        # カーネル変換
        K = gower_to_kernel(
            self.distance_matrix_,
            method=self.kernel_method,
            sigma=self.kernel_sigma,
        )
        
        # SS-KNMFの学習
        self.model.fit(K, labeled_indices, labels)
        
        return self
    
    def predict(self) -> np.ndarray:
        """クラスタラベルを予測
        
        Returns:
            クラスタラベル (n,)
        """
        return self.model.predict()
    
    def get_cluster_membership(self) -> np.ndarray:
        """クラスタ所属確率を取得
        
        Returns:
            所属度行列 (k×n)
        """
        return self.model.get_cluster_membership()
    
    def get_reconstruction_errors(self) -> List[float]:
        """再構成誤差の履歴を取得
        
        Returns:
            各反復での再構成誤差のリスト
        """
        return self.model.get_reconstruction_errors()
