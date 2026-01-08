import os
import logging

from itertools import product
from typing import get_args, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans

from lib.data import EncodeMethod, NormalizeMethod, encode_categorical, normalize
from lib.cluster_index import ClusterIndex
from dataset.utils import load_dataset, sampling


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    datasets: list[str]
    n_samples: int
    cluster_method: str
    seed: int = 42
    debug: bool = False


def normal_clustering(df, with_label=True, n_clusters: int = 5):
    if n_clusters < 5:
        raise ValueError("n_clusters must be greater than 5")

    score = ClusterIndex(with_label=with_label)

    x = df.drop("Label").to_numpy()
    y = df["Label"].to_numpy()

    start = n_clusters - 4
    end = n_clusters + 5

    for n_clusters in range(start, end):
        # logger.info(f"n_clusters={n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(x)
        score.add(n_clusters, x, cluster_labels, y, kmeans)

    return score


class Cluster:
    def _get_cluster_method(self):
        method_name = self.config.cluster_method.lower()
        if method_name == "kmeans":
            return normal_clustering
        else:
            raise ValueError(f"Unknown cluster method: {method_name}")

    def __init__(self, config: ClusterConfig):
        self.config = config

        self.cluster_method = self._get_cluster_method()
        self.dataset_index = 0
        self.df_sampling = None
        self.metadata: Optional[dict] = None

    def _load_dataset(self):
        df, metadata = load_dataset(
            self.config.datasets[self.dataset_index],
            self.config.debug
        )
        df = sampling(df, self.config.n_samples, self.config.seed)

        category_columns = []
        file = os.path.join(
            os.path.dirname(__file__),
            f"dataset_metadata/{self.config.datasets[self.dataset_index]}.txt"
        )
        with open(file, "r") as f:
            for line in f:
                category_columns.append(line.strip())

        category_columns = [col for col in category_columns if col in df.columns]
        metadata["category_columns"] = category_columns

        return df, metadata

    def _loop(self, encode_method, normalize_method):
        if self.df_sampling is None:
            raise ValueError("df_sampling is None")
        df = self.df_sampling.clone()

        cols_before = len(df.columns)
        if self.metadata is None:
            raise ValueError("metadata is None")

        logger.info(f"Method: encode->{encode_method}, normalize->{normalize_method}")

        df = encode_categorical(
            df,
            self.metadata.get("category_columns", []),
            method=encode_method
        )
        cols_after = len(df.columns)
        cols_diff = cols_after - cols_before

        logger.info(f"  Columns: {cols_before} -> {cols_after} (diff: {cols_diff:+d})")

        df = normalize(
            df,
            self.metadata.get("category_columns", []),
            method=normalize_method
        )
        score = normal_clustering(df, n_clusters=int(self.metadata.get("n_clusters", 0)))
        # logger.info(score)

    def run(self):
        self.df_sampling, self.metadata = self._load_dataset()

        encode_method = get_args(EncodeMethod)
        normalize_methods = get_args(NormalizeMethod)

        for encode_method, normalize_method in product(encode_method, normalize_methods):
            self._loop(encode_method, normalize_method)
