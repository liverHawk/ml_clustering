import argparse
import time
from pathlib import Path
import polars as pl

from itertools import product
from typing import get_args

from lib.data import encode_categorical, EncodeMethod, NormalizeMethod, normalize, make_sample_data
from dataset.cicids2017 import relabeled_dataset
from main import normal_clustering, save_score


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def load_dataset(dataset_name: str = "CICIDS2017_improved", debug: bool = False):
    path = Path(f"/Users/toshi_pro/Documents/school/dataset/project/cleaned/{dataset_name}")
    files = list(path.glob("*.csv"))
    dfs = []
    for file in files:
        df_tmp = pl.read_csv(file)
        dfs.append(df_tmp)
    else:
        del df_tmp
    df =  pl.concat(dfs)
    df = df.drop(["Src IP", "Dst IP", "Timestamp"])

    if debug:
        for col in df.columns:
            print(col)
        print(df["Label"].value_counts())
    df = relabeled_dataset(df)
    n_labels = df["Label"].n_unique()
    return df, { "dataset": dataset_name, "n_clusters": n_labels }


def _main():
    args = load_args()
    time_string = f'{time.strftime("%Y%m%d-%H%M%S")}'
    df_original, metadata = load_dataset(args.dataset, args.debug)

    df_sampling = df_original.group_by("Label").agg(
        pl.all().sample(n=5, seed=42)
    ).explode(
        pl.all().exclude("Label")
    )

    encode_methods = get_args(EncodeMethod)
    normalize_methods = get_args(NormalizeMethod)

    for category_method, normalize_method in product(encode_methods, normalize_methods):
        df_copy = df_sampling.clone()
        print(f"{category_method} {normalize_method}")
        df = encode_categorical(df_copy, ["Src Port", "Dst Port", "Protocol"], method=category_method)
        df = normalize(df, ["Src Port", "Dst Port", "Protocol"], method=normalize_method)
        score = normal_clustering(df, n_clusters=metadata["n_clusters"])
        save_score(score, category_method, normalize_method, timing=time_string)

    with open(f'./results/csv/{time_string}/metadata.txt', 'w') as f:
        for k, v in metadata.items():
            f.write(f'{k}: {v}\n')


if __name__ == "__main__":
    _main()