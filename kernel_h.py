import comet_ml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clustering_methods.configs import GowerSSKNMFConfig
from clustering_methods import GowerSSKNMF
from lib.cluster_index import ClusterIndex


import yaml
import argparse
import logging
import coloredlogs
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def load_params():
    path = Path(__file__).parent / "params.yaml"
    params = yaml.load(path.read_text(), Loader=yaml.FullLoader)
    
    # 数値型に変換（YAMLでは科学記法が文字列になることがある）
    if "tol" in params:
        params["tol"] = float(params["tol"])
    if "alpha" in params:
        params["alpha"] = float(params["alpha"])
    if "n_samples_per_label" in params:
        params["n_samples_per_label"] = int(params["n_samples_per_label"])
    if "labeled_rate" in params:
        params["labeled_rate"] = float(params["labeled_rate"])
    if "random_state" in params:
        params["random_state"] = int(params["random_state"])
    if "max_iter" in params:
        params["max_iter"] = int(params["max_iter"])
    if "kernel_sigma" in params:
        params["kernel_sigma"] = float(params["kernel_sigma"])
    
    # 制約手法のパラメータ
    if "constraint_method" not in params:
        params["constraint_method"] = "hard"
    if "beta" in params:
        params["beta"] = float(params["beta"])
    if "learning_rate" in params:
        params["learning_rate"] = float(params["learning_rate"])
    if "alpha_init" in params and params["alpha_init"] is not None:
        params["alpha_init"] = float(params["alpha_init"])
    if "alpha_final" in params and params["alpha_final"] is not None:
        params["alpha_final"] = float(params["alpha_final"])
    if "confidence_weights" in params and params["confidence_weights"] is not None:
        params["confidence_weights"] = np.array(params["confidence_weights"])
    
    return params


def _plot_confusion_matrix(true_and_predictions, save_path, n_clusters):
    true_labels = true_and_predictions["true_labels"].to_numpy()
    predictions = true_and_predictions["predictions"].to_numpy()
    
    # ユニークなラベルを取得
    unique_true_labels = sorted(set(true_labels))
    unique_predictions = sorted(set(predictions))
    
    # ラベルからインデックスへのマッピングを作成
    true_label_to_idx = {label: idx for idx, label in enumerate(unique_true_labels)}
    pred_to_idx = {pred: idx for idx, pred in enumerate(unique_predictions)}
    
    # マトリックスを初期化
    cm = np.zeros((len(unique_true_labels), len(unique_predictions)), dtype=int)
    
    # カウント
    for true_label, pred in zip(true_labels, predictions):
        cm[true_label_to_idx[true_label], pred_to_idx[pred]] += 1
    
    # ラベル名を準備
    yticklabels = unique_true_labels
    xticklabels = [f"Cluster {p}" for p in unique_predictions]
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="Blues"
    )
    plt.title(f"Confusion Matrix (n_clusters={n_clusters})")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Label")
    plt.savefig(save_path / f"confusion_matrix_{n_clusters:02d}.png", dpi=300, bbox_inches='tight')
    plt.close()


def load_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--datasets", type=str, default="all")
    parser.add_argument("-d", "--dataset", type=str, default="CICIDS2017_flow_improved")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-n", "--n_samples_per_label", type=int, default=100)
    parser.add_argument("-l", "--labeled_rate", type=float, default=0.4)
    parser.add_argument("-r", "--random_state", type=int, default=42)
    args = parser.parse_args()
    # args.datasets = [ds.strip() for ds in args.datasets.split(",")]
    return args


def main():
    exp = comet_ml.start(project_name="cluster")
    save_path = Path(f'./results')
    save_path.mkdir(parents=True, exist_ok=True)

    # args = load_args()
    params = load_params()

    center_n_clusters = len(params["use_labels"])  # = len(use_labels)
    start = max(center_n_clusters - 4, len(params["known_labels"]))
    end = center_n_clusters + 5

    exp.add_tags(params["tags"] + ["constraint_method"])
    exp.log_parameters({
        **params,
        "n_clusters_range": list(range(start, end))
    })

    config = GowerSSKNMFConfig(
        base_path="/home/toshi/Documents/dataset/project/cleaned",
        dataset_name=params["dataset"],
        n_samples_per_label=params["n_samples_per_label"],
        labeled_rate=params["labeled_rate"],
        random_state=params["random_state"]
    )
    model = GowerSSKNMF(config)
    logger.info(model.get_labels())
    model.set_labels(
        use_labels=params["use_labels"],
        known_labels=params["known_labels"]
    )
    model.set_cols(
        categorical_columns=params["categorical_columns"]
    )
    kernel_sigma = params.get("kernel_sigma", None)
    kernel_method = params.get("kernel_method", "rbf")
    model.convert_to_kernel(kernel_method=kernel_method, kernel_sigma=kernel_sigma)
    
    score = ClusterIndex(with_label=True)

    # 制約手法のパラメータを取得
    constraint_method = params.get("constraint_method", "hard")
    beta = params.get("beta", 0.8)
    learning_rate = params.get("learning_rate", 0.2)
    confidence_weights = params.get("confidence_weights", None)
    alpha_init = params.get("alpha_init", None)
    alpha_final = params.get("alpha_final", None)
    
    logger.info(f"制約手法: {constraint_method}")
    if constraint_method == "interpolation":
        logger.info(f"  beta: {beta}")
    elif constraint_method == "partial":
        logger.info(f"  learning_rate: {learning_rate}")
    elif constraint_method == "adaptive_alpha":
        logger.info(f"  alpha_init: {alpha_init}, alpha_final: {alpha_final}")

    for n_clusters in range(start, end):
        logger.info(f"n_clusters: {n_clusters}")
        predictions, membership = model.predict(
            n_clusters, 
            params["alpha"], 
            params["max_iter"], 
            params["tol"],
            constraint_method=constraint_method,
            beta=beta,
            learning_rate=learning_rate,
            confidence_weights=confidence_weights,
            alpha_init=alpha_init,
            alpha_final=alpha_final
        )

        # logger.info(predictions)
        # logger.info(membership)

        evaluation = pl.DataFrame({
            "kernel": model.kernel,
            "predictions": predictions,
            "true_labels": model.df_setup["Label"].to_list(),
        })

        _plot_confusion_matrix(evaluation, save_path, n_clusters)

        score.add(
            n_clusters,
            evaluation["kernel"].to_numpy(),
            evaluation["predictions"].to_numpy(),
            evaluation["true_labels"].to_numpy(),
        )
        exp.log_metrics(score.get_results(n_clusters), step=n_clusters)

    score.plot(
        path=save_path / "kernel_class_",
        separate_plots=False
    )
    score.save_data(
        path=save_path
    )
    for file in save_path.glob("*.png"):
        exp.log_image(file)


if __name__ == "__main__":
    main()
