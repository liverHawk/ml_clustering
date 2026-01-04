import polars as pl
from typing import Literal, List
from sklearn.datasets import make_blobs
import numpy as np

from .general import create_centers_with_distances

EncodeMethod = Literal['one-hot', 'label']
NormalizeMethod = Literal['z-score', 'minmax', 'robust', 'none']

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


def encode_categorical(df: pl.DataFrame, columns: List[str], method: EncodeMethod = 'one-hot'):
    match method:
        case 'one-hot':
            return df.to_dummies(columns)
        case 'label':
            for col in columns:
                unique_values = df[col].unique().to_list()
                mapping_dict = {
                    val: idx for idx, val in enumerate(unique_values)
                }
                # print(mapping_dict)
                df = df.with_columns(
                    pl.col(col).replace(mapping_dict).cast(pl.Int8).alias(f'{col}_encoded')
                )
                df = df.drop(col)
            return df
        case 'none':
            return df
        case _:
            raise ValueError('Invalid encoding method')


def normalize(df: pl.DataFrame, except_original_columns: List[str], method: NormalizeMethod = 'z-score'):
    if method == 'none':
        return df

    except_columns = ['Label']
    for col in df.columns:
        for include in except_original_columns:
            if include in col:
                except_columns.append(col)
                break

    except_df = df[except_columns]
    df = df.drop(except_columns)
    # print(except_df.columns, df.columns)
    df_normalized = pl.DataFrame()
    col = ''
    try:
        match method:
            case 'z-score':
                df_normalized = df.with_columns([
                    (
                        (pl.col(col) - pl.col(col).mean()) / (pl.col(col).std() + 1e-6)
                    ).alias(col) for col in df.columns
                ])
            case 'minmax':
                df_normalized = df.with_columns([
                    (
                        (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min() + 1e-6)
                    ).alias(col) for col in df.columns
                ])
            case 'robust':
                df_normalized = df.with_columns([
                    (
                        (pl.col(col) - pl.col(col).median()) / (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25) + 1e-6)
                    ).alias(col) for col in df.columns
                ])
            case _:
                raise ValueError('Invalid normalization method')
    except Exception as e:
        print(e, col)

    df = pl.concat([df_normalized, except_df], how='horizontal')
    return df
