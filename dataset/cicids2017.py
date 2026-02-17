import polars as pl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def relabeled_dataset(df: pl.DataFrame, convert_labels: bool = True):
    # Polars の with_columns/filter は元の DataFrame を変更しない（immutable）ため clone() は不要
    labels = df['Label'].unique().to_list()
    for label in labels:
        # logger.info(label)
        if "Attempted" in label:
            df = df.with_columns(
                pl.when(pl.col("Label") == label)
                .then(pl.lit("BENIGN"))
                .otherwise(pl.col("Label"))
                .alias("Label")
            )
        if not convert_labels:
            continue
        if "Web Attack" in label:
            df = df.with_columns(
                pl.when(pl.col("Label") == label)
                .then(pl.lit("Web Attack"))
                .otherwise(pl.col("Label"))
                .alias("Label")
            )
        if "DoS" in label and "DDoS" not in label:
            df = df.with_columns(
                pl.when(pl.col("Label") == label)
                .then(pl.lit("DoS"))
                .otherwise(pl.col("Label"))
                .alias("Label")
            )
        if "Portscan" in label:
            df = df.with_columns(
                pl.when(pl.col("Label") == label)
                .then(pl.lit("Portscan"))
                .otherwise(pl.col("Label"))
                .alias("Label")
            )
    
    # remove row which has "Label" is "BENIGN"
    df = df.filter(pl.col("Label") != "BENIGN")

    return df