import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import argparse
import logging
import coloredlogs

from clustering_methods.gower_ssknmf import gower_distance_vectorized, gower_to_kernel
from clustering_methods import SSKNMF, SSKNMFConfig
from dataset.utils import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


BASE_PATH = "/home/toshi/Documents/dataset/project/cleaned"

USE_LABELS = [
    "DoS",
    "Botnet",
    "Infiltration",
    "Web Attack",
    "SSH-Patator",
    "Heartbleed",
    "DDoS",
    "FTP-Patator",
    "Portscan",
    "Heartbleed",
]

KNOWN_LABELS = [
    "DoS",
    "SSH-Patator",
    "Web Attack",
]

N_SAMPLES_PER_LABEL = 100

LABELED_RATE = 0.5

CATEGORICAL_COLS = [
    "Src Port",
    "Dst Port",
    "Protocol",
]


def load_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--datasets", type=str, default="all")
    parser.add_argument("-d", "--dataset", type=str, default="CICIDS2017_flow_improved")
    parser.add_argument("-v", "--verbose", action='store_true')
    # parser.add_argument("-n", "--n_samples", type=int, default=0)
    args = parser.parse_args()
    # args.datasets = [ds.strip() for ds in args.datasets.split(",")]
    return args


def main():
    args = load_args()

    df, metadata = load_dataset(args.dataset, base_path=BASE_PATH)
    # logger.info(f"dataset DataFrame: {df.shape}")
    # logger.info(f"all labels: {df['Label'].unique().to_list()}")

    all_labels = df["Label"].unique().to_list()
    use_labels = [label for label in USE_LABELS if label in all_labels]
    df: pl.DataFrame = df.filter(pl.col("Label").is_in(use_labels))
    # logger.info(f"use labels: {use_labels}(len: {len(use_labels)})")

    df = df.with_row_index("index")

    df_known: pl.DataFrame = df.filter(pl.col("Label").is_in(KNOWN_LABELS))
    df_unknown: pl.DataFrame = df.filter(~pl.col("Label").is_in(KNOWN_LABELS))
    # logger.info(f"df_known: {df_known.shape}")
    # logger.info(f"df_unknown: {df_unknown.shape}")

    # =============== known ====================
    df_sample = df_known.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(n=N_SAMPLES_PER_LABEL, seed=42)
    )
    # logger.info(f"df_sample: {df_sample.shape}")

    labeled_samples = df_sample.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(n=int(N_SAMPLES_PER_LABEL * LABELED_RATE), seed=42)
    )
    # logger.info(f"labeled_indices: len: {len(labeled_indices)}")

    label_mapping = {label: i for i, label in enumerate(KNOWN_LABELS)}
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
    no_labeled_samples = no_labeled_samples.with_columns(
        pl.lit("-1").cast(label_encoded_dtype).alias("label_encoded")
    )


    # ----- labeled_indices and labels -----

    # ================= unknown ====================
    unknown_samples = df_unknown.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(n=min(N_SAMPLES_PER_LABEL, len(group)), seed=42)
    )
    unknown_samples = unknown_samples.with_columns(
        pl.lit("-1").cast(label_encoded_dtype).alias("label_encoded")
    )
    # ----- unlabeled_samples -----

    df_combined = pl.concat([
        no_labeled_samples,
        labeled_samples,
        unknown_samples
    ])
    df_combined = df_combined.drop("index").with_row_index("index")

    labeled = df_combined.filter(pl.col("label_encoded") != "-1")
    labeled_indices = labeled["index"].to_list()
    labels = labeled["label_encoded"].to_list()

    # logger.info(df_combined.columns)

    # labeled_indices and labels
    delete_columns = ["index", "Label", "label_encoded"]

    numerical_cols = [col for col in df_combined.columns if col not in CATEGORICAL_COLS + delete_columns]
    # logger.info(f"numerical_cols: {numerical_cols}")

    distance_matrix = gower_distance_vectorized(
        df_combined,
        CATEGORICAL_COLS,
        numerical_cols
    )
    # logger.info(f"distance_matrix: {distance_matrix.shape}")

    kernel_matrix = gower_to_kernel(
        distance_matrix,
        method="rbf"
    )
    assert np.all(kernel_matrix >= 0) and np.all(kernel_matrix <= 1)
    """
    ここまでをGowerSSKNMF.__init__でしたい
    """
    # logger.info(f"kernel_matrix: {kernel_matrix.shape}")

    n_clusters = len(use_labels)

    config = SSKNMFConfig(
        n_clusters=n_clusters,
        alpha=0.5,
        max_iter=100,
        tol=1e-4,
        random_state=42,
        verbose=True
    )
    model = SSKNMF(config)

    logger.info("fitting...")
    model.fit(kernel_matrix, labeled_indices, labels)
    logger.info("fitting done")

    logger.info("predicting...")
    predictions = model.predict()
    logger.info(f"predictions: {predictions}")
    logger.info("predicting done")

    logger.info("getting membership...")
    membership = model.get_cluster_membership()
    logger.info(f"membership: {membership}")

    # ================ evaluation ====================
    # ----- prepare -----
    # add column "predicted_label" to df_combined (from predictions)
    df_combined = df_combined.with_columns(
        pl.Series(predictions).alias("predicted_label")
    )
    # ----- confusion matrix -----
    cm_rows = []
    for true_label in use_labels:
        row = {"True Label": true_label}
        # 各予測ラベルに対するカウント
        counts = (
            df_combined
            .filter(pl.col("Label") == true_label)
            .group_by("predicted_label")
            .agg(pl.len().alias("count"))
            .sort("predicted_label")
        )
        
        # 全ての予測ラベルに対してカウントを設定
        for pred_label in range(n_clusters):
            count_row = counts.filter(pl.col("predicted_label") == pred_label)
            row[f"Cluster {pred_label}"] = count_row["count"][0] if len(count_row) > 0 else 0
        
        cm_rows.append(row)
    cm_df = pl.DataFrame(cm_rows)
    # logger.info(f"cm_dict\n{cm_dict}")

    plt.figure(figsize=(10, 8))
    # 数値データのみを抽出（True Label列を除く）
    cm_values = cm_df.select([col for col in cm_df.columns if col != "True Label"]).to_numpy()
    true_labels = cm_df["True Label"].to_list()
    predicted_labels = [col for col in cm_df.columns if col != "True Label"]
    ax = sns.heatmap(
        cm_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=predicted_labels,
        yticklabels=true_labels
    )
    # label_mappingで指定されているラベル（KNOWN_LABELS）に対応するy軸ラベルを太文字にする
    yticklabels = ax.get_yticklabels()
    for i, label in enumerate(true_labels):
        if label in KNOWN_LABELS:
            yticklabels[i].set_weight('bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    
    
if __name__ == "__main__":
    main()
