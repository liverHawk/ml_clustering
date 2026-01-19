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

from dataset.utils import load_dataset

from .configs import SSKNMFConfig, GowerSSKNMFConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gower_distance_vectorized(
    df: pl.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    categorical_weight: float = 1.0,
    numerical_weight: float = 1.0,
) -> np.ndarray:
    """ベクトル化されたGower距離計算
    
    Gower距離は混合型データ（カテゴリ変数と数値変数）に対して
    統一的に距離を計算する手法です。
    
    Args:
        df: Polars DataFrame
        categorical_cols: カテゴリ変数のカラム名リスト
        numerical_cols: 数値変数のカラム名リスト
        categorical_weight: カテゴリ変数の重み（デフォルト: 1.0）
        numerical_weight: 数値変数の重み（デフォルト: 1.0）
    
    Returns:
        Gower距離行列 (n×n)
    
    Note:
        計算量: O(n² * f)
        メモリ: O(n²)
    """
    n = len(df)
    n_cat = len(categorical_cols)
    n_num = len(numerical_cols)
    n_features = n_cat + n_num
    
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
        num_dist *= numerical_weight
    
    # カテゴリ変数の距離計算
    cat_dist = np.zeros((n, n))
    if len(categorical_cols) > 0:
        cat_data = df.select(categorical_cols).to_numpy()
        
        # ブロードキャスト: (n,1,f) != (1,n,f) → (n,n,f)
        for k in range(len(categorical_cols)):
            cat_dist += (cat_data[:, None, k] != cat_data[None, :, k]).astype(float)
        cat_dist *= categorical_weight
    
    # 重み付き正規化
    total_weight = n_cat * categorical_weight + n_num * numerical_weight
    if total_weight == 0:
        raise ValueError("Total weight must be greater than 0")
    
    return (num_dist + cat_dist) / total_weight


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
        max_iter = self.config.max_iter
        
        for iteration in range(1, max_iter + 1):
            H_old = self.H_.copy()
            
            # Hの更新（反復回数を渡す）
            self._update_H(K, H_constraint, unlabeled_mask, labeled_indices, iteration, max_iter)
            
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
        
        if iteration == max_iter and self.config.verbose:
            logger.info("最大反復回数に達しました")
    
    def _update_H(
        self,
        K: np.ndarray,
        H_constraint: Optional[np.ndarray],
        unlabeled_mask: np.ndarray,
        labeled_indices: Optional[np.ndarray],
        iteration: int = 1,
        max_iter: int = 200,
    ):
        """H行列の更新
        
        更新則:
        H ← H ⊙ √[(KH^T + αH_constraint)^T / (HH^TKH^T + αH)^T]
        
        Args:
            K: カーネル行列
            H_constraint: 制約行列
            unlabeled_mask: ラベルなしサンプルのマスク
            labeled_indices: ラベル付きサンプルのインデックス
            iteration: 現在の反復回数
            max_iter: 最大反復回数
        """
        assert self.H_ is not None, "H must be initialized"
        
        # 適応的alphaの計算（制約手法がadaptive_alphaの場合）
        current_alpha = self.config.alpha
        if self.config.constraint_method == 'adaptive_alpha':
            alpha_init = self.config.alpha_init if self.config.alpha_init is not None else self.config.alpha
            alpha_final = self.config.alpha_final if self.config.alpha_final is not None else self.config.alpha * 0.1
            # 線形に減少
            progress = (iteration - 1) / (max_iter - 1) if max_iter > 1 else 0.0
            current_alpha = alpha_init * (1 - progress) + alpha_final * progress
        
        # 中間計算
        KH_T = K @ self.H_.T  # (n×k)
        HH_T = self.H_ @ self.H_.T  # (k×k)
        HH_TKH_T = HH_T @ KH_T.T  # (k×n)
        
        # 分子: (KH^T)^T + α * H_constraint
        numerator = KH_T.T  # (k×n)
        if H_constraint is not None and current_alpha > 0:
            numerator += current_alpha * H_constraint
        
        # 分母: HH^TKH^T + α * H
        denominator = HH_TKH_T  # (k×n)
        if current_alpha > 0:
            denominator += current_alpha * self.H_
        
        denominator = np.maximum(denominator, self.config.eps)
        
        # 全サンプルに対して更新式を計算
        H_updated = self.H_ * np.sqrt(numerator / denominator)
        
        # ラベルなしサンプルは常に更新
        self.H_[:, unlabeled_mask] = H_updated[:, unlabeled_mask]
        
        # ラベル付きサンプルの更新方法を選択
        if labeled_indices is not None and H_constraint is not None:
            method = self.config.constraint_method.lower()
            
            if method == 'hard':
                # 方法0: 完全固定（元の方法）
                self.H_[:, labeled_indices] = H_constraint[:, labeled_indices]
            
            elif method == 'soft':
                # 方法1: ソフト制約（更新式に制約項を含める）
                self.H_[:, labeled_indices] = H_updated[:, labeled_indices]
            
            elif method == 'interpolation':
                # 方法2: 重み付き混合
                beta = self.config.beta
                self.H_[:, labeled_indices] = (
                    (1 - beta) * H_updated[:, labeled_indices] + 
                    beta * H_constraint[:, labeled_indices]
                )
            
            elif method == 'partial':
                # 方法3: 部分的更新
                lr = self.config.learning_rate
                self.H_[:, labeled_indices] = (
                    self.H_[:, labeled_indices] + 
                    lr * (H_updated[:, labeled_indices] - self.H_[:, labeled_indices])
                )
            
            elif method == 'relaxation':
                # 方法4: 反復的な制約緩和
                # 初期は強く（beta=1.0）、後期は緩和（beta=0.5）
                progress = (iteration - 1) / (max_iter - 1) if max_iter > 1 else 0.0
                beta = 1.0 * (1 - progress) + 0.5 * progress
                self.H_[:, labeled_indices] = (
                    (1 - beta) * H_updated[:, labeled_indices] + 
                    beta * H_constraint[:, labeled_indices]
                )
            
            elif method == 'confidence':
                # 方法5: 信頼度ベースの制約
                if self.config.confidence_weights is not None:
                    # confidence_weightsは各ラベル付きサンプルに対する信頼度
                    conf_weights = self.config.confidence_weights
                    if len(conf_weights) != len(labeled_indices):
                        raise ValueError(f"confidence_weights length {len(conf_weights)} must match labeled_indices length {len(labeled_indices)}")
                    
                    # 各サンプルに対して異なるbetaを適用
                    for i, idx in enumerate(labeled_indices):
                        beta_i = 1.0 - conf_weights[i]  # 信頼度が高いほどbetaが小さい（制約が弱い）
                        self.H_[:, idx] = (
                            (1 - beta_i) * H_updated[:, idx] + 
                            beta_i * H_constraint[:, idx]
                        )
                else:
                    # 信頼度が指定されていない場合は通常のinterpolationと同じ
                    beta = self.config.beta
                    self.H_[:, labeled_indices] = (
                        (1 - beta) * H_updated[:, labeled_indices] + 
                        beta * H_constraint[:, labeled_indices]
                    )
            
            elif method == 'adaptive_alpha':
                # 方法6: 制約項の重み調整（alphaは既に上で調整済み）
                # ソフト制約として更新
                self.H_[:, labeled_indices] = H_updated[:, labeled_indices]
            
            else:
                # デフォルトはhard制約
                logger.warning(f"Unknown constraint_method: {method}, using 'hard'")
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
    def __init__(self, config: GowerSSKNMFConfig):
        self.config = config

        self.df_original, self.metadata = load_dataset(
            dataset_name=self.config.dataset_name,
            debug=self.config.debug,
            base_path=self.config.base_path
        )
        self.all_labels = self.df_original["Label"].unique().to_list()
        self.use_labels = None
        self.known_labels = None
        self.use_known_no_labeled = False

        self.categorical_cols = None

        self.df_setup = None
        self.labeled_indices = None
        self.labels = None
        self.kernel = None
        
        # 特徴量の重み（デフォルト値）
        self.categorical_weight = 1.0
        self.numerical_weight = 1.0

    def _setup(self):
        use_labels = [label for label in self.use_labels if label in self.all_labels]
        df: pl.DataFrame = self.df_original.filter(pl.col("Label").is_in(use_labels))
        # logger.info(f"use labels: {use_labels}(len: {len(use_labels)})")

        df = df.with_row_index("index")

        df_known: pl.DataFrame = df.filter(pl.col("Label").is_in(self.known_labels))
        df_unknown: pl.DataFrame = df.filter(~pl.col("Label").is_in(self.known_labels))
        # logger.info(f"df_known: {df_known.shape}")
        # logger.info(f"df_unknown: {df_unknown.shape}")

        # =============== known ====================
        # known_labelsが空の場合は、ラベル付きサンプルなしで処理
        if len(self.known_labels) == 0 or len(df_known) == 0:
            # ラベル付きサンプルなしの場合
            label_encoded_dtype = pl.Int32  # デフォルトの型
            # スキーマをコピーしてlabel_encoded列を追加
            schema_dict = dict(df.schema)
            schema_dict["label_encoded"] = label_encoded_dtype
            labeled_samples = pl.DataFrame(schema=schema_dict)
            no_labeled_samples = pl.DataFrame(schema=schema_dict)
            _labeled_indices = []
            labels = []
            logger.info("known_labelsが空のため、ラベル付きサンプルなしで処理します")
        else:
            df_sample = df_known.group_by("Label", maintain_order=True).map_groups(
                lambda group: group.sample(n=min(self.config.n_samples_per_label, len(group)), seed=42)
            )
            # logger.info(f"df_sample: {df_sample.shape}")

            labeled_samples = df_sample.group_by("Label", maintain_order=True).map_groups(
                lambda group: group.sample(n=min(int(self.config.n_samples_per_label * self.config.labeled_rate), len(group)), seed=42)
            )
            # logger.info(f"labeled_indices: len: {len(labeled_indices)}")

            label_mapping = {label: i for i, label in enumerate(self.known_labels)}
            logger.info(f"label_mapping: {label_mapping}")
            labeled_samples = labeled_samples.with_columns(
                pl.col("Label").replace(label_mapping).alias("label_encoded")
            )
            _labeled_indices = labeled_samples["index"].to_list()
            label_encoded_dtype = labeled_samples["label_encoded"].dtype
            # logger.info(labeled_samples["label_encoded"].head(3))
            # logger.info(labeled_samples[['Label', 'label_encoded']].group_by("Label").head(3))
            labels = labeled_samples["label_encoded"].to_list()

            no_labeled_samples = df_sample.filter(~pl.col("index").is_in(_labeled_indices))
            # label_encoded_dtypeが数値型の場合は-1、文字列型の場合は"-1"を使用
            if label_encoded_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                no_labeled_samples = no_labeled_samples.with_columns(
                    pl.lit(-1).cast(label_encoded_dtype).alias("label_encoded")
                )
            else:
                no_labeled_samples = no_labeled_samples.with_columns(
                    pl.lit("-1").cast(label_encoded_dtype).alias("label_encoded")
                )

        # ----- labeled_indices and labels -----

        # ================= unknown ====================
        if len(df_unknown) > 0:
            unknown_samples = df_unknown.group_by("Label", maintain_order=True).map_groups(
                lambda group: group.sample(n=min(self.config.n_samples_per_label, len(group)), seed=42)
            )
            # label_encoded_dtypeが数値型の場合は-1、文字列型の場合は"-1"を使用
            if label_encoded_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                unknown_samples = unknown_samples.with_columns(
                    pl.lit(-1).cast(label_encoded_dtype).alias("label_encoded")
                )
            else:
                unknown_samples = unknown_samples.with_columns(
                    pl.lit("-1").cast(label_encoded_dtype).alias("label_encoded")
                )
        else:
            # 空のDataFrameを作成（label_encoded列を含む）
            # スキーマをコピーしてlabel_encoded列を追加
            schema_dict = dict(df.schema)
            schema_dict["label_encoded"] = label_encoded_dtype
            unknown_samples = pl.DataFrame(schema=schema_dict)
        # ----- unlabeled_samples -----

        # データフレームの結合
        dfs_to_concat = []
        if self.use_known_no_labeled and len(no_labeled_samples) > 0:
            dfs_to_concat.append(no_labeled_samples)
        if len(labeled_samples) > 0:
            dfs_to_concat.append(labeled_samples)
        if len(unknown_samples) > 0:
            dfs_to_concat.append(unknown_samples)
        
        if len(dfs_to_concat) > 0:
            df_combined = pl.concat(dfs_to_concat)
        else:
            raise ValueError("結合するデータフレームがありません。use_labelsとknown_labelsの設定を確認してください。")
        df_combined = df_combined.drop("index").with_row_index("index")

        # label_encodedの型に応じて比較値を変更
        # 数値型の場合は-1、文字列型の場合は"-1"
        if label_encoded_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            labeled = df_combined.filter(pl.col("label_encoded") != -1)
        else:
            labeled = df_combined.filter(pl.col("label_encoded") != "-1")
        labeled_indices = labeled["index"].to_list()
        labels = labeled["label_encoded"].to_list()

        return df_combined, labeled_indices, labels
    
    def get_labels(self) -> List[str]:
        return self.df_original["Label"].unique().to_list()
    
    def get_columns(self) -> List[str]:
        return self.df_original.columns.to_list()
    
    def set_labels(self, use_labels: List[str], known_labels: List[str], use_known_no_labeled: bool = False):
        self.use_labels = use_labels
        self.known_labels = known_labels
        self.use_known_no_labeled = use_known_no_labeled
        
        self.df_setup, self.labeled_indices, self.labels = self._setup()

    def set_cols(self, categorical_columns: List[str]):
        self.categorical_cols = categorical_columns
    
    def set_feature_weights(
        self, 
        categorical_weight: float = 1.0,
        numerical_weight: float = 1.0
    ):
        """特徴量の重みを設定
        
        Args:
            categorical_weight: カテゴリ変数の重み（大きいほど重要）
            numerical_weight: 数値変数の重み
        """
        if categorical_weight < 0 or numerical_weight < 0:
            raise ValueError("Weights must be non-negative")
        self.categorical_weight = categorical_weight
        self.numerical_weight = numerical_weight

    def convert_to_kernel(self, kernel_method: str = "rbf", kernel_sigma: float = None):
        assert self.categorical_cols is not None
        
        # 除外する列を定義（存在する列のみ）
        exclude_cols = ["index", "Label", "label_encoded"]
        cols_to_drop = [col for col in exclude_cols if col in self.df_setup.columns]
        
        numerical_cols = [
            col for col in self.df_setup.columns 
            if col not in self.categorical_cols + cols_to_drop
        ]

        distance_matrix = gower_distance_vectorized(
            self.df_setup.drop(cols_to_drop),
            self.categorical_cols,
            numerical_cols,
            categorical_weight=self.categorical_weight,
            numerical_weight=self.numerical_weight
        )

        kernel_matrix = gower_to_kernel(
            distance_matrix,
            method=kernel_method,
            sigma=kernel_sigma,
        )
        assert np.all(kernel_matrix >= 0) and np.all(kernel_matrix <= 1)

        self.kernel = kernel_matrix

    def get_kernel(self) -> np.ndarray:
        return self.kernel

    def predict(self, n_clusters, alpha, max_iter, tol, 
                constraint_method='hard', beta=0.8, learning_rate=0.2,
                confidence_weights=None, alpha_init=None, alpha_final=None):
        config = SSKNMFConfig(
            n_clusters=n_clusters,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=self.config.random_state,
            verbose=self.config.debug,
            constraint_method=constraint_method,
            beta=beta,
            learning_rate=learning_rate,
            confidence_weights=confidence_weights,
            alpha_init=alpha_init,
            alpha_final=alpha_final
        )
        model = SSKNMF(config)
        model.fit(self.kernel, self.labeled_indices, self.labels)

        assert self.kernel is not None
        assert self.labeled_indices is not None

        predictions = model.predict()
        membership = model.get_cluster_membership()

        return predictions, membership

