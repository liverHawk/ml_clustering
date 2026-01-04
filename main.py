import polars as pl
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from itertools import product
from typing import get_args

from lib.cluster_index import ClusterIndex
from lib.data import encode_categorical, EncodeMethod, NormalizeMethod, normalize
from lib.general import create_centers_with_distances

def make_sample_data():
    n_samples = 1000
    n_features = 80

    centers = create_centers_with_distances(
        n_clusters=5,
        n_features=n_features,
        distances=[0.1, 1, 1.1, 10, 100]
    )

    x, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        random_state=42
    )

    regions = ["Tokyo", "Osaka", "Kyoto", "Nagoya", "Fukuoka", "Sapporo"]
    ranks = ["Gold", "Silver", "Bronze", "Platinum", "Diamond", "Master"]

    rng = np.random.default_rng(42)
    random_regions = rng.choice(regions, size=n_samples)
    random_ranks = rng.choice(ranks, size=n_samples)

    df = pl.DataFrame({
        **{ f"feature_{i}": x[:, i] for i in range(n_features) },
        "region": random_regions,
        "rank": random_ranks,
        "Label": y,
    })
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": len(centers),
    }
    return df, metadata


def normal_clustering(df, with_label=True):
    score = ClusterIndex(with_label=with_label)
    x = df.drop("Label").to_numpy()
    y = df["Label"].to_numpy()

    for n_clusters in range(2, 10):
        # print(f"n_clusters={n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(x)
        score.add(n_clusters, x, cluster_labels, y, kmeans)

    return score


def save_score(score: ClusterIndex, method, normalize_method: NormalizeMethod = 'none', timing: str = None):
    os.makedirs(f'./results/csv/{timing}', exist_ok=True)
    file_name = f'{timing}/{method}_{normalize_method}'
    with open(f'./results/csv/{file_name}.csv', 'w') as f:
        result = score.get_results()
        cluster_keys = list(result.keys())
        value_keys = list(result[cluster_keys[0]].keys())
        f.write('n_clusters,' + ','.join(value_keys) + '\n')
        for n_clusters in cluster_keys:
            f.write(f'{n_clusters},' + ','.join(
                str(result[n_clusters][k]) for k in value_keys
            ) + '\n')

def main():
    time_string = f'{time.strftime("%Y%m%d-%H%M%S")}'
    df_original, metadata = make_sample_data()

    encode_methods = get_args(EncodeMethod)
    normalize_methods = get_args(NormalizeMethod)

    for category_method, normalize_method in product(encode_methods, normalize_methods):
        df_copy = df_original.clone()
        print(f"{category_method} {normalize_method}")
        df = encode_categorical(df_copy, ["region", "rank"], method=category_method)
        df = normalize(df, ["region", "rank"], method=normalize_method)
        score = normal_clustering(df)
        save_score(score, category_method, normalize_method, timing=time_string)

    with open(f'./results/csv/{time_string}/metadata.txt', 'w') as f:
        for k, v in metadata.items():
            f.write(f'{k}: {v}\n')


if __name__ == "__main__":
    main()