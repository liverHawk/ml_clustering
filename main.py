import os
import time
import logging
import coloredlogs

from sklearn.cluster import KMeans
from itertools import product
from typing import get_args

from lib.cluster_index import ClusterIndex
from lib.data import encode_categorical, EncodeMethod, NormalizeMethod, normalize, make_sample_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


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

def _main():
    time_string = f'{time.strftime("%Y%m%d-%H%M%S")}'
    df_original, metadata = make_sample_data()

    encode_methods = get_args(EncodeMethod)
    normalize_methods = get_args(NormalizeMethod)

    for category_method, normalize_method in product(encode_methods, normalize_methods):
        df_copy = df_original.clone()
        logger.info(f"{category_method} {normalize_method}")
        df = encode_categorical(df_copy, ["region", "rank"], method=category_method)
        df = normalize(df, ["region", "rank"], method=normalize_method)
        score = normal_clustering(df)
        save_score(score, category_method, normalize_method, timing=time_string)

    with open(f'./results/csv/{time_string}/metadata.txt', 'w') as f:
        for k, v in metadata.items():
            f.write(f'{k}: {v}\n')


if __name__ == "__main__":
    _main()