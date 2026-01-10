import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import polars as pl
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass
from typing import Optional, List

from clustering_methods import GowerSSKNMF
from dataset.utils import load_dataset
from lib.data import get_methods, normalize
from lib.cluster_index import ClusterIndex
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SSKNMFConfig:
    dataset_name: str = "CICIDS2017_improved"
    dataframe: Optional[pl.DataFrame] = None
    categorical_cols: Optional[List[str]] = None
    exclude_labels: Optional[List[str]] = None
    n_samples_per_label: int = 5
    n_clusters: Optional[int] = None
    kernel_method: str = 'rbf'
    kernel_sigma: Optional[float] = None
    alpha: float = 0.4
    max_iter: int = 200
    tol: float = 1e-4
    verbose: bool = True
    random_state: int = 42
    seed: int = 42
    debug: bool = False


class SSKNMFCluster:
    def __init__(self, config: SSKNMFConfig):
        self.config = config
        self.df_raw: Optional[pl.DataFrame] = None
        self.df_combined: Optional[pl.DataFrame] = None
        self.metadata: Optional[dict] = None
        self.model: Optional[GowerSSKNMF] = None
        self.cluster_labels: Optional[np.ndarray] = None
        
    def _load_dataset(self):
        """データセットを読み込む"""
        self.df_raw, self.metadata = load_dataset(
            self.config.dataset_name,
            self.config.debug
        )
        
        if self.config.categorical_cols is None:
            self.config.categorical_cols = ["Src Port", "Dst Port", "Protocol"]
        
        if self.config.exclude_labels is None:
            self.config.exclude_labels = ["BENIGN", "DDoS", "Heartbleed", "Infiltration", "Botnet", "SSH-Patator", "Web Attack"]
        
        if self.config.n_clusters is None:
            self.config.n_clusters = self.df_raw["Label"].n_unique()
            logger.info(f"n_clusters not specified, using number of unique labels: {self.config.n_clusters}")
        else:
            logger.info(f"n_clusters specified: {self.config.n_clusters}")
        
        logger.info(f"Loaded dataset: {self.config.dataset_name}")
        logger.info(f"Total samples: {len(self.df_raw)}")
        logger.info(f"Number of unique labels: {self.df_raw['Label'].n_unique()}")
        logger.info(f"Number of clusters: {self.config.n_clusters}")
    
    def _sample_labeled_data(self, df_with_index, unique_labels):
        """ラベル付きデータをサンプリング"""
        labeled_dfs = []
        for label in unique_labels:
            if label in self.config.exclude_labels:
                continue
            label_df = df_with_index.filter(pl.col("Label") == label)
            n_samples = min(self.config.n_samples_per_label, len(label_df))
            if n_samples > 0:
                sampled = label_df.sample(n=n_samples, seed=self.config.seed)
                labeled_dfs.append(sampled)
        return pl.concat(labeled_dfs) if labeled_dfs else pl.DataFrame()
    
    def _sample_unlabeled_data(self, df_with_index, labeled_df):
        """ラベルなしデータをサンプリング"""
        selected_no_labeled = df_with_index.filter(
            ~pl.col("row_index").is_in(labeled_df["row_index"].to_list())
        )
        return selected_no_labeled.group_by("Label", maintain_order=True).map_groups(
            lambda group: group.sample(n=self.config.n_samples_per_label, seed=self.config.seed)
        )
    
    def _prepare_data(self):
        """データのサンプリングとラベル準備"""
        df_with_index = self.df_raw.with_row_index("row_index")
        unique_labels = self.df_raw["Label"].unique().to_list()
        label_map = {label: i for i, label in enumerate(unique_labels)} # 既知攻撃だけでは
        print(f"unique_labels: {label_map}")
        
        labeled_df = self._sample_labeled_data(df_with_index, unique_labels)
        sampled_no_labeled = self._sample_unlabeled_data(df_with_index, labeled_df)
        
        self.df_combined = pl.concat([sampled_no_labeled, labeled_df])
        
        n_unlabeled = len(sampled_no_labeled)
        labeled_indices = np.arange(n_unlabeled, n_unlabeled + len(labeled_df))
        labels = labeled_df["Label"].replace(label_map).to_numpy()
        
        logger.info(f"Combined data: {len(self.df_combined)} samples")
        logger.info(f"Labeled samples: {len(labeled_indices)}")
        
        return labeled_indices, labels, unique_labels, label_map
    
    def _create_model(self):
        """モデルを作成"""
        numerical_cols = [
            col for col in self.df_raw.columns 
            if col not in self.config.categorical_cols + ["Label"]
        ]
        
        self.model = GowerSSKNMF(
            categorical_cols=self.config.categorical_cols,
            numerical_cols=numerical_cols,
            n_clusters=self.config.n_clusters,
            kernel_method=self.config.kernel_method,
            kernel_sigma=self.config.kernel_sigma,
            alpha=self.config.alpha,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            verbose=self.config.verbose,
            random_state=self.config.random_state
        )
    
    def _create_confusion_matrix(self, combined_labels, unique_labels, n_clusters):
        """combined_labelsから混同行列を作成"""
        combined_labels = combined_labels.with_columns(
            pl.col("Predicted Cluster").cast(pl.Utf8).alias("Predicted Cluster")
        )
        
        confusion_df = (
            combined_labels
            .group_by(["True Label", "Predicted Cluster"], maintain_order=True)
            .agg(pl.len().alias("Count"))
        )
        
        cluster_cols = [str(i) for i in range(n_clusters)]
        cm_df = confusion_df.pivot(
            index="True Label",
            on="Predicted Cluster",
            values="Count",
            aggregate_function="sum"
        ).fill_null(0)
        
        for col in cm_df.columns:
            if col != "True Label" and col.isdigit():
                cm_df = cm_df.rename({col: f"Cluster {col}"})
        
        predicted_label_names = [f"Cluster {i}" for i in range(n_clusters)]
        for col in predicted_label_names:
            if col not in cm_df.columns:
                cm_df = cm_df.with_columns(pl.lit(0).alias(col))
        
        cm_df = cm_df.select(["True Label"] + predicted_label_names)
        
        label_to_order = {label: i for i, label in enumerate(unique_labels)}
        cm_df = cm_df.with_columns(
            pl.col("True Label").map_elements(
                lambda x: label_to_order.get(x, len(unique_labels)),
                return_dtype=pl.Int32
            ).alias("_order")
        ).sort("_order").drop("_order")
        
        existing_labels = set(cm_df["True Label"].to_list())
        missing_labels = [label for label in unique_labels if label not in existing_labels]
        if missing_labels:
            missing_data = {"True Label": missing_labels}
            for col in predicted_label_names:
                if col in cm_df.columns:
                    col_type = cm_df[col].dtype
                    missing_data[col] = pl.Series([0] * len(missing_labels), dtype=col_type)
                else:
                    missing_data[col] = [0] * len(missing_labels)
            missing_rows = pl.DataFrame(missing_data)
            cm_df = pl.concat([cm_df, missing_rows])
            cm_df = cm_df.with_columns(
                pl.col("True Label").map_elements(
                    lambda x: label_to_order.get(x, len(unique_labels)),
                    return_dtype=pl.Int32
                ).alias("_order")
            ).sort("_order").drop("_order")
        
        cm = cm_df.select(predicted_label_names).to_numpy()
        true_label_names = cm_df["True Label"].to_list()
        
        return cm, true_label_names, predicted_label_names
    
    def _plot_confusion_matrix(self, cm, true_label_names, predicted_label_names, 
                               normalized=False, save_path=None):
        """混同行列のヒートマップを表示"""
        if normalized:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot = np.divide(
                cm.astype('float'), 
                row_sums, 
                out=np.zeros_like(cm, dtype=float), 
                where=(row_sums != 0)
            ) * 100
            fmt = '.1f'
            cbar_label = 'Percentage (%)'
            title = 'Confusion Matrix Heatmap (Normalized): True Label vs Predicted Cluster'
            if save_path is None:
                save_path = 'confusion_matrix_heatmap_normalized.png'
        else:
            cm_plot = cm
            fmt = 'd'
            cbar_label = 'Sample Count'
            title = 'Confusion Matrix Heatmap: True Label vs Predicted Cluster'
            if save_path is None:
                save_path = 'confusion_matrix_heatmap.png'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=predicted_label_names,
            yticklabels=true_label_names,
            cbar_kws={'label': cbar_label}
        )
        plt.xlabel('Predicted Cluster', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix heatmap saved to '{save_path}'")
        plt.show()
    
    def _evaluate_clustering(self, cluster_labels, true_labels):
        """ClusterIndexを使って評価値を計算"""
        score = ClusterIndex(with_label=True)
        
        # 特徴量行列を取得
        x = self.df_combined.drop("Label").drop("row_index").to_numpy()
        
        # 距離行列を使用してsilhouette_scoreを計算
        if self.model.distance_matrix_ is not None:
            silhouette = silhouette_score(
                self.model.distance_matrix_,
                cluster_labels,
                metric='precomputed'
            )
        else:
            # 距離行列がない場合は特徴量から計算
            silhouette = silhouette_score(x, cluster_labels)
        
        # ClusterIndexのaddメソッドを使用
        # kmeansがNoneの場合、_wb_indexでエラーになる可能性があるため、
        # 内部評価指標を手動で計算
        if len(np.unique(cluster_labels)) >= 2:
            from sklearn.metrics import (
                calinski_harabasz_score,
                davies_bouldin_score
            )
            ch_score = calinski_harabasz_score(x, cluster_labels)
            db_score = davies_bouldin_score(x, cluster_labels)
            
            score.results[self.config.n_clusters] = {
                "silhouette_score": silhouette,
                "ch_score": ch_score,
                "db_score": db_score
            }
        
        # 外部評価指標（ARI, NMI, FMI）
        if score.with_label:
            from sklearn.metrics import (
                adjusted_rand_score,
                normalized_mutual_info_score,
                fowlkes_mallows_score
            )
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            fmi = fowlkes_mallows_score(true_labels, cluster_labels)
            
            score.results_with_label[self.config.n_clusters] = {
                "ARI": ari,
                "NMI": nmi,
                "FMI": fmi
            }
        
        return score
    
    def run(self):
        """メイン処理を実行"""
        self._load_dataset()
        labeled_indices, labels, unique_labels, label_map = self._prepare_data()
        self._create_model()
        
        self.cluster_labels = self.model.fit_predict(
            self.df_combined.drop("row_index"),
            labeled_indices,
            labels
        )
        
        true_labels = self.df_combined["Label"].to_numpy()
        
        # 評価値の計算
        score = self._evaluate_clustering(self.cluster_labels, true_labels)
        results = score.get_results()
        
        logger.info(f"\n評価結果 (n_clusters={self.config.n_clusters}):")
        for metric, value in results[self.config.n_clusters].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        combined_labels = pl.DataFrame({
            "True Label": true_labels, 
            "Predicted Cluster": self.cluster_labels
        })
        
        cm, true_label_names, predicted_label_names = self._create_confusion_matrix(
            combined_labels, unique_labels, self.config.n_clusters
        )
        
        self._plot_confusion_matrix(cm, true_label_names, predicted_label_names, normalized=False)
        self._plot_confusion_matrix(cm, true_label_names, predicted_label_names, normalized=True)
        
        return self.cluster_labels, score



# def ssknmf_clustering(df):
    

def single_ssknmf(dataset, verbose, n_samples):
    df_original, metadata = load_dataset(dataset, debug=verbose)
    least_label_count = df_original["Label"].value_counts()["count"].min()

    if n_samples == 0:
        n_samples = least_label_count
    elif least_label_count < n_samples:
        raise ValueError(f"Least label count is less than n_samples: {least_label_count} < {n_samples}")
    
    df_sampling = df_original.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(n=n_samples, seed=42)
    )
    metadata["n_samples"] = len(df_sampling)
    metadata["n_samples_per_label"] = n_samples

    category_columns, _, normalize_methods = get_methods(df_sampling)

    for normalize_method in normalize_methods:
        df_copy = df_sampling.clone()
        df = normalize(df_copy, category_columns, method=normalize_method)

        score = ssknmf_clustering(df, n_clusters=metadata["n_clusters"])

        save_score(score, normalize_method, timing=time_string)

def _main():
    args = load_args()

    if args.datasets == "all":
        pass
    else:
        for dataset in args.datasets:
            single_ssknmf(dataset, args.verbose, args.n_samples)

if __name__ == "__main__":
    config = SSKNMFConfig(
        dataset_name="CICIDS2017_improved",
        n_clusters=9,  # クラスタ数を明示的に指定（Noneの場合は自動設定）
        alpha=0.8,
        verbose=True
    )
    
    cluster = SSKNMFCluster(config)
    cluster_labels, score = cluster.run()
