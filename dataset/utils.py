from polars.lazyframe import LazyFrame
import polars as pl
import logging

from pathlib import Path

from lib.data import get_schema
from . import cicids2017


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str = "CICIDS2017_improved", debug: bool = False):
    path = Path(f"/home/hawk/Documents/school/dataset/project/cleaned/{dataset_name}")
    files = list(path.glob("*.csv"))
    schema = get_schema(files)
    # 遅延評価でメモリ効率を向上（スキーマを事前に指定して型推論を回避）
    dfs: list[LazyFrame] = [pl.scan_csv(file, schema_overrides=schema) for file in files]
    df = pl.concat(dfs).collect()

    delete_columns = ["Src IP", "Dst IP", "Timestamp", "Source IP", "Destination IP", "SimillarHTTP"]
    delete_columns = [col for col in delete_columns if col in df.columns]

    df = df.drop(delete_columns)

    if debug:
        for col in df.columns:
            logger.info(col)
        value_counts = df["Label"].value_counts()
        value_counts.write_csv("./results/csv/value_counts.csv")

    if dataset_name in ["CICIDS2017_improved", "CICIDS2017_flow_improved", "CSECICIDS2018_improved"]:
        df = cicids2017.relabeled_dataset(df)
    elif dataset_name in ["CICDDoS2019"]:
        # df = cicddos2019.relabeled_dataset(df)
        pass
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n_labels = df["Label"].n_unique()

    return df, { "dataset": dataset_name, "n_clusters": n_labels }


def sampling(df: pl.DataFrame, n_samples: int = 5, seed: int = 42):
    value_count_min: int = df["Label"].value_counts()["count"].min()  # ty:ignore[invalid-assignment]
    if n_samples == 0:
        n_samples = value_count_min
    elif value_count_min < n_samples:
        raise ValueError(f"Least label count is less than n_samples: {value_count_min} < {n_samples}")

    df_sampling = df.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(n=n_samples, seed=seed)
    )
    return df_sampling
