import argparse
import logging
import coloredlogs

import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--times", type=str, required=True)
    parser.add_argument("-o", "--overwrite", action='store_true')
    args = parser.parse_args()
    args.times = [time.strip() for time in args.times.split(",")]
    return args


def single_analyze(analyze_time: str, overwrite: bool = False):
    path = Path(f"./results/csv/{analyze_time}")

    analyze_path = path / "analyze.png"
    if analyze_path.exists() and not overwrite:
        return
    wb_index_path = path / "wb_index.png"
    if wb_index_path.exists() and not overwrite:
        return

    logger.info("--------------------------------")
    logger.info(f"Analyzing time: {analyze_time}")

    files = path.glob("*.csv")

    n_columns = 3
    fig, axes = plt.subplots(
        3,
        n_columns,
        figsize=(24, 15)
    )

    score_name_list = [
        "silhouette_score", "ch_score", "db_score",
        "ARI", "NMI", "FMI", 'wb_index'
    ]
    mapping = {}

    for idx, score_name in enumerate(score_name_list):
        mapping[score_name] = idx
        row = idx // n_columns
        col = idx % n_columns
        axes[row, col].set_title(score_name)
        axes[row, col].set_xlabel("n_clusters")
        axes[row, col].set_ylabel("Score")
        axes[row, col].grid(True)


    for file in files:
        df = pl.read_csv(file)
        n_clusters = df['n_clusters'].to_list()

        file_name = file.stem

        for score_name in score_name_list:
            idx = mapping[score_name]
            axes[idx // n_columns, idx % n_columns].plot(
                n_clusters,
                df[score_name].to_list(),
                label=file_name
            )

    for idx in mapping.values():
        axes[idx // n_columns, idx % n_columns].legend()

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
    plt.tight_layout()
    plt.savefig(analyze_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    files = path.glob("*.csv")
    for file in files:
        df = pl.read_csv(file)
        plt.plot(df['n_clusters'].to_list(), df['wb_index'].to_list(), label=file.stem)
        min_val = min(df['wb_index'].to_list())
        min_pos = df['wb_index'].to_list().index(min_val)
        min_n = df['n_clusters'].to_list()[min_pos]
        plt.scatter(min_n, min_val, color='red')
        plt.annotate(f"min: {min_val:.2f} at n={min_n}", (min_n, min_val), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title("WB Index")
    plt.xlabel("n_clusters")
    plt.ylabel("WB Index")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(wb_index_path)
    plt.close()
    
    return

def main():
    args = load_args()

    if len(args.times) == 1 and args.times[0] == "all":
        path = Path("./results/csv/processed_datasets.csv")
        df_time = pl.read_csv(path, columns=["time"])
        times = df_time["time"].to_list()
        for time in times:
            try:
                single_analyze(time, args.overwrite)
            except Exception as e:
                logger.error(f"Error analyzing time: {time}")
                logger.error(e)
                continue
    else:
        for time in args.times:
            try:
                single_analyze(time, args.overwrite)
            except Exception as e:
                logger.error(f"Error analyzing time: {time}")
                logger.error(e)
                continue
    


if __name__ == "__main__":
    main()



