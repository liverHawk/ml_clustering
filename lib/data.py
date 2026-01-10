import polars as pl
from typing import Literal, List
from sklearn.datasets import make_blobs
import numpy as np

from .general import create_centers_with_distances
import logging
from typing import get_args


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EncodeMethod = Literal['one-hot', 'label']
NormalizeMethod = Literal['z-score', 'minmax', 'robust', 'none']


def get_methods(df: pl.DataFrame):
    category_columns = ["Source Port", "Destination Port", "Protocol", "Src Port", "Dst Port"]
    category_columns = [col for col in category_columns if col in df.columns]
    encode_methods = get_args(EncodeMethod)
    normalize_methods = get_args(NormalizeMethod)
    return category_columns, encode_methods, normalize_methods


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
                # logger.info(mapping_dict)
                df = df.with_columns(
                    # Int8 だとユニーク値が多いときにオーバーフローするので、余裕を持って Int32 にする
                    pl.col(col).replace(mapping_dict).cast(pl.Int32).alias(f'{col}_encoded')
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
    # logger.info(except_df.columns, df.columns)
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
        logger.info(e, col)

    df = pl.concat([df_normalized, except_df], how='horizontal')
    return df


def get_schema(files):
    unified_schema = {}
    for file in files:
        # logger.info(f"Getting schema for {file}")
        current_schema = pl.read_csv(file, n_rows=0, schema_overrides={ "SimillarHTTP": pl.Utf8 }).schema
        for col, dtype in current_schema.items():
            if col =="SimillarHTTP":
                unified_schema[col] = pl.Utf8
                continue
            if col not in unified_schema:
                unified_schema[col] = dtype
            else:
                if unified_schema[col] != dtype:
                    if dtype.is_float() or unified_schema[col].is_float():
                        unified_schema[col] = pl.Float64

    return unified_schema
