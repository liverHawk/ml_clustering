import argparse
import polars as pl
import logging
import coloredlogs

from pathlib import Path
from itertools import product
from typing import List, Literal

from lib.data import get_schema, EncodeMethod, NormalizeMethod, get_args, encode_categorical, normalize
from dataset.cicids2017 import relabeled_dataset

ClusterMethod = Literal['kmeans', 'dbscan', 'hierarchical', 'spectral', 'affinity', 'meanshift', 'optics', 'birch', 'gmm']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datasets", type=str, required=True)
    parser.add_argument("-n", "--n_samples", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    # カンマ区切りの文字列をリストに変換
    args.datasets = [ds.strip() for ds in args.datasets.split(",")]
    return args


def load_dataset(dataset_name: str = "CICIDS2017_improved", debug: bool = False):
    path = Path(f"/home/hawk/Documents/school/dataset/project/cleaned/{dataset_name}")
    files = list(path.glob("*.csv"))
    schema = get_schema(files)
    # 遅延評価でメモリ効率を向上（スキーマを事前に指定して型推論を回避）
    dfs = [pl.scan_csv(file, schema_overrides=schema) for file in files]
    df = pl.concat(dfs).collect()

    delete_columns = ["Src IP", "Dst IP", "Timestamp", "Source IP", "Destination IP", "SimillarHTTP"]
    delete_columns = [col for col in delete_columns if col in df.columns]

    df = df.drop(delete_columns)

    if debug:
        for col in df.columns:
            logger.info(col)
        value_counts = df["Label"].value_counts()
        value_counts.write_csv("./results/csv/value_counts.csv")

    df = relabeled_dataset(df)
    n_labels = df["Label"].n_unique()
    return df, { "dataset": dataset_name, "n_clusters": n_labels }


def sampling(df: pl.DataFrame, n_samples: int = 5, seed: int = 42):
    if n_samples == 0:
        n_samples = df["Label"].value_counts()["count"].min()
    elif df["Label"].value_counts()["count"].min() < n_samples:
        raise ValueError(f"Least label count is less than n_samples: {df['Label'].value_counts()['count'].min()} < {n_samples}")

    df_sampling = df.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(n=n_samples, seed=seed)
    )
    return df_sampling


def loop(df_sampling: pl.DataFrame, category_columns: List[str], category_method: EncodeMethod = 'one-hot', normalize_method: NormalizeMethod = 'z-score', cluster_method: ClusterMethod = 'kmeans', debug: bool = False):
    df_copy = df_sampling.clone()
    df = encode_categorical(df_copy, category_columns, method=category_method)
    df = normalize(df, category_columns, method=normalize_method)

    if cluster_method == 'kmeans':
        score = normal_clustering(df, n_clusters=metadata["n_clusters"])
        return score


def dataset_method(dataset: str, n_samples: int = 5, seed: int = 42):
    logger.info(dataset)
    df, metadata = load_dataset(dataset)
    df_sampling = sampling(df, n_samples, seed)
    logger.info(df_sampling.shape)
    logger.info(metadata)

    category_columns = ["Source Port", "Destination Port", "Protocol", "Src IP", "Dst IP"]
    category_columns = [col for col in category_columns if col in df_sampling.columns]
    encode_methods = get_args(EncodeMethod)
    normalize_methods = get_args(NormalizeMethod)

    for category_method, normalize_method in product(encode_methods, normalize_methods):
        loop(df_sampling, category_columns, category_method, normalize_method)


def main():
    args = load_args()
    for dataset in args.datasets:
        dataset_method(dataset, args.n_samples, args.seed)


if __name__ == "__main__":
    main()