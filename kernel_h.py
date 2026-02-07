import comet_ml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clustering_methods.configs import GowerSSKNMFConfig
from clustering_methods import GowerSSKNMF
from lib.cluster_index import ClusterIndex
from dataset.utils import load_dataset
from lib.experiment_db import ClusterResult, Experiment, get_session

import yaml
import argparse
import logging
import coloredlogs
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

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
    if "categorical_weight" in params:
        params["categorical_weight"] = float(params["categorical_weight"])
    if "numerical_weight" in params:
        params["numerical_weight"] = float(params["numerical_weight"])
    if "normalize_numerical" in params and params["normalize_numerical"] is not None:
        params["normalize_numerical"] = str(params["normalize_numerical"]).strip().lower()
    
    if "base_path" in params:
        params["base_path"] = str(params["base_path"]).strip()
    if "relabel" in params:
        params["relabel"] = bool(params["relabel"])
    
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
    
    # リスト型のパラメータを正規化（文字列の場合はリストに変換）
    if "known_labels" in params:
        if isinstance(params["known_labels"], str):
            params["known_labels"] = [params["known_labels"]]
        elif not isinstance(params["known_labels"], list):
            params["known_labels"] = list(params["known_labels"])
    
    if "use_labels" in params:
        if isinstance(params["use_labels"], str):
            params["use_labels"] = [params["use_labels"]]
        elif not isinstance(params["use_labels"], list):
            params["use_labels"] = list(params["use_labels"])
    
    if "exclude_labels" in params:
        if isinstance(params["exclude_labels"], str):
            params["exclude_labels"] = [params["exclude_labels"]]
        elif not isinstance(params["exclude_labels"], list):
            params["exclude_labels"] = list(params["exclude_labels"])
    
    if "categorical_columns" in params:
        if isinstance(params["categorical_columns"], str):
            params["categorical_columns"] = [params["categorical_columns"]]
        elif not isinstance(params["categorical_columns"], list):
            params["categorical_columns"] = list(params["categorical_columns"])
    
    if "tags" in params:
        if isinstance(params["tags"], str):
            params["tags"] = [params["tags"]]
        elif not isinstance(params["tags"], list):
            params["tags"] = list(params["tags"])
    
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
    base_path = params["base_path"]

    df_original, metadata = load_dataset(params["dataset"], debug=False, base_path=base_path, relabel=params["relabel"])

    if params["exclude_labels"] and len(params["exclude_labels"]) > 0:
        df_original = df_original.filter(~pl.col("Label").is_in(params["exclude_labels"]))

    n_clusters = df_original["Label"].n_unique()
    all_labels = df_original["Label"].unique().to_list()
    with open(f"dataset_metadata/{params['dataset']}_label.txt", "w") as f:
        for label in all_labels:
            f.write(f"{label}\n")


    center_n_clusters = len(params["use_labels"])  # = len(use_labels)
    if center_n_clusters == 0:
        center_n_clusters = n_clusters
    start = max(center_n_clusters - 4, len(params["known_labels"]) + 1, 1)
    end = center_n_clusters + 5

    exp.add_tags(params["tags"] + ["constraint_method"])
    exp.log_parameters({
        **params,
        "n_clusters_range": list(range(start, end))
    })

    config = GowerSSKNMFConfig(
        base_path=base_path,
        dataset_name=params["dataset"],
        n_samples_per_label=params["n_samples_per_label"],
        labeled_rate=params["labeled_rate"],
        random_state=params["random_state"]
    )
    model = GowerSSKNMF(config)
    logger.info(model.get_labels())
    model.set_labels(
        use_labels=params["use_labels"],
        known_labels=params["known_labels"],
        use_known_no_labeled=params["use_known_no_labeled"]
    )
    model.set_cols(
        categorical_columns=params["categorical_columns"]
    )
    
    # 特徴量の重みを設定
    categorical_weight = params.get("categorical_weight", 1.0)
    numerical_weight = params.get("numerical_weight", 1.0)
    model.set_feature_weights(
        categorical_weight=categorical_weight,
        numerical_weight=numerical_weight
    )
    logger.info(f"特徴量の重み: categorical_weight={categorical_weight}, numerical_weight={numerical_weight}")
    
    normalize_numerical = params.get("normalize_numerical", "none")
    model.set_normalize_numerical(normalize_numerical)
    logger.info(f"Gower距離計算前の数値列正規化: {normalize_numerical}")
    
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

    # DB セッションと Experiment レコードの作成
    session = get_session()
    try:
        try:
            comet_key = exp.get_key()
        except Exception:
            comet_key = None

        experiment = Experiment(
            dataset_name=params["dataset"],
            constraint_method=constraint_method,
            kernel_method=kernel_method,
            tags=json.dumps(params.get("tags", []), ensure_ascii=False),
            params_json=json.dumps(params, default=str, ensure_ascii=False),
            n_clusters_start=start,
            n_clusters_end=end - 1,
            comet_experiment_key=comet_key,
        )
        session.add(experiment)
        session.flush()  # experiment.id を取得する

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

            # _plot_confusion_matrix(evaluation, save_path, n_clusters)

            # 既知ラベルが固定されたクラスタIDを取得（0からlen(known_labels)-1まで）
            known_cluster_ids = list(range(len(params["known_labels"])))
            
            # 既知ラベルが固定されたクラスタに割り当てられたデータを除外
            evaluation_filtered = evaluation.filter(
                ~pl.col("predictions").is_in(known_cluster_ids)
            )
            
            logger.info(
                f"既知ラベル固定クラスタ ({known_cluster_ids}) のデータを除外: "
                f"全データ数={len(evaluation)}, 評価対象データ数={len(evaluation_filtered)}"
            )
            
            # 除外後のデータで評価指標を計算
            if len(evaluation_filtered) > 0:
                score.add(
                    n_clusters,
                    evaluation_filtered["kernel"].to_numpy(),
                    evaluation_filtered["predictions"].to_numpy(),
                    evaluation_filtered["true_labels"].to_numpy(),
                )

                metrics = score.get_results(n_clusters)
                exp.log_metrics(metrics, step=n_clusters)

                result = ClusterResult(
                    experiment_id=experiment.id,
                    n_clusters=n_clusters,
                    silhouette_score=metrics.get("silhouette_score"),
                    ch_score=metrics.get("ch_score"),
                    db_score=metrics.get("db_score"),
                    ARI=metrics.get("ARI"),
                    NMI=metrics.get("NMI"),
                    FMI=metrics.get("FMI"),
                )
                session.add(result)
            else:
                logger.warning(
                    f"n_clusters={n_clusters}: 評価対象データが0件のため、評価指標をスキップします"
                )

        session.commit()
    finally:
        session.close()

    # score.plot(
    #     path=save_path / "kernel_class_",
    #     separate_plots=False
    # )
    # score.save_data(
    #     path=save_path
    # )
    # for file in save_path.glob("*.png"):
    #     exp.log_image(file)


if __name__ == "__main__":
    main()
