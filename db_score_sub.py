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


def single_db_analyze(time: str, overwrite: bool = False):
    path = Path(f"./results/csv/{time}")

    db_score_acceleration_path = path / "db_score_acceleration.png"
    if db_score_acceleration_path.exists() and not overwrite:
        # logger.info("DB Score Acceleration file already exists. Skipping...")
        return

    logger.info("--------------------------------")
    logger.info(f"Analyzing time: {time}")

    files = sorted(list(path.glob("*.csv")))

    if not files:
        logger.info("No CSV files found.")
        return

    # 1枚の図に「スコア」「1次階差（変化量）」「2次階差（変化量の変化）」の3つを並べる
    fig, axes = plt.subplots(
        len(files),
        3,
        figsize=(24, 5 * len(files)),
        squeeze=False
    )

    for i, file in enumerate(files):
        df = pl.read_csv(file)
        n_clusters = df['n_clusters'].to_list()
        db_scores = df['db_score'].to_list()

        # 1次階差 (Velocity): スコアの差
        db_diff = []

        n_clusters_diff = []
        for j in range(1, len(db_scores) - 1):
            val = db_scores[j + 1] + db_scores[j - 1] - 2 * db_scores[j]
            db_diff.append(val)
            n_clusters_diff.append(n_clusters[j])

        # 2次階差 (Acceleration): 変化量の差
        # これが正に大きく振れる地点が、減少が止まった（カーブが曲がった）「エルボー点」
        db_accel = []
        n_clusters_accel = []
        for j in range(1, len(db_diff) - 1):
            val = db_diff[j + 1] + db_diff[j - 1] - 2 * db_diff[j]
            db_accel.append(val)
            n_clusters_accel.append(n_clusters_diff[j])

        file_name = file.stem

        # 1. スコアそのもの
        axes[i, 0].plot(n_clusters, db_scores, marker='o', color='tab:blue')
        axes[i, 0].set_title(f"DB Score - {file_name}")
        axes[i, 0].grid(True)

        # 2. 1次階差 (変化量 g(n))
        axes[i, 1].bar(n_clusters_diff, db_diff, color='tab:orange', alpha=0.6)
        axes[i, 1].axhline(0, color='black', lw=1, ls='--')
        axes[i, 1].set_title("1st Derivative (f(n+1) + f(n-1))")
        axes[i, 1].grid(True, axis='y')

        # 3. 2次階差 (変化量の変化 h(n))
        axes[i, 2].bar(n_clusters_accel, db_accel, color='tab:red', alpha=0.6)
        axes[i, 2].axhline(0, color='black', lw=1, ls='--')
        axes[i, 2].set_title("2nd Derivative (g(n+1) - g(n-1))")
        axes[i, 2].grid(True, axis='y')

        # 最も「変化の勢いが弱まった」地点を特定（エルボー候補）
        if db_accel:
            # db's elbow -> 下に凸 -> V字
            max_accel = max(db_accel)
            max_idx = db_accel.index(max_accel)
            target_n = n_clusters[max_idx + 2]  # index調整
            axes[i, 2].annotate(
                f'Elbow Point?\n(n={target_n})',
                xy=(target_n, max_accel),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center',
                color='red',
                arrowprops=dict(arrowstyle='->', color='red')
            )

    plt.tight_layout()
    plt.savefig(db_score_acceleration_path)
    plt.close()

def main():
    args = load_args()

    if len(args.times) == 1 and args.times[0] == "all":
        path = Path("./results/csv/processed_datasets.csv")
        df_time = pl.read_csv(path, columns=["time"])
        times = df_time["time"].to_list()
        for time in times:
            try:
                single_db_analyze(time, args.overwrite)
            except Exception as e:
                logger.error(f"Error analyzing time: {time}")
                logger.error(e)
                continue
    else:
        for time in args.times:
            try:
                single_db_analyze(time, args.overwrite)
            except Exception as e:
                logger.error(f"Error analyzing time: {time}")
                logger.error(e)
                continue


if __name__ == "__main__":
    main()