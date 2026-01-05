import argparse
import time
from pathlib import Path
import polars as pl
import logging
import coloredlogs

from itertools import product
from typing import get_args

from lib.data import encode_categorical, EncodeMethod, NormalizeMethod, normalize, get_schema
from dataset.cicids2017 import relabeled_dataset
from main import normal_clustering, save_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

coloredlogs.install(level='INFO')


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datasets", type=str, required=True)
    parser.add_argument("-n", "--n_samples", type=int, default=5)
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


def single(dataset, n_samples: int = 0, debug: bool = False):
    logger.info("--------------------------------")
    logger.info(f"Processing dataset: {dataset}")
    time_string = f'{time.strftime("%Y%m%d-%H%M%S")}'
    df_original, metadata = load_dataset(dataset, debug)

    least_label_count = df_original["Label"].value_counts()["count"].min()

    if n_samples == 0:
        n_samples = least_label_count
    elif least_label_count < n_samples:
        raise ValueError(f"Least label count is less than n_samples: {least_label_count} < {n_samples}")

    df_sampling = df_original.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(n=n_samples, seed=42)
    )
    metadata["n_samples"] = len(df_sampling)
    
    category_columns = ["Source Port", "Destination Port", "Protocol", "Src IP", "Dst IP"]
    category_columns = [col for col in category_columns if col in df_sampling.columns]
    encode_methods = get_args(EncodeMethod)
    normalize_methods = get_args(NormalizeMethod)

    for category_method, normalize_method in product(encode_methods, normalize_methods):
        df_copy = df_sampling.clone()
        logger.info(f"{category_method} {normalize_method}")
        
        # エンコード前の列数を記録
        cols_before = len(df_copy.columns)
        df = encode_categorical(df_copy, category_columns, method=category_method)
        cols_after = len(df.columns)
        cols_diff = cols_after - cols_before
        
        logger.info(f"  Columns: {cols_before} -> {cols_after} (diff: {cols_diff:+d})")
        
        df = normalize(df, category_columns, method=normalize_method)
        score = normal_clustering(df, n_clusters=metadata["n_clusters"])
        save_score(score, category_method, normalize_method, timing=time_string)

    with open(f'./results/csv/{time_string}/metadata.txt', 'w') as f:
        for k, v in metadata.items():
            f.write(f'{k}: {v}\n')
    
    path = "./results/csv/processed_datasets.csv"
    if not Path(path).exists():
        with open(path, "w") as f:
            f.write("time,dataset\n")

    with open(path, "a") as f:
        f.write(f"{time_string},{dataset}\n")


def _main():
    args = load_args()

    if len(args.datasets) == 1 and args.datasets[0] == "all":
        path = Path("/home/hawk/Documents/school/dataset/project/cleaned")
        # search all directories in path
        datasets = [dir.name for dir in path.iterdir() if dir.is_dir()]
        
        logger.info("--------------------------------")
        logger.info(f"Found {len(datasets)} datasets")
        for dataset in datasets:
            logger.info(dataset)
        
        args.datasets = datasets
    
    for dataset in args.datasets:
        try:
            single(dataset, args.n_samples, args.debug)
        except Exception as e:
            logger.error(f"Error processing dataset: {dataset}")
            logger.error(e)
            continue

    logger.info("Finished all datasets")

if __name__ == "__main__":
    _main()