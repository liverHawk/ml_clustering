import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt

ANALYZE_TIME = '20260104-160822'

def main():
    path = Path(f"./results/csv/{ANALYZE_TIME}")
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
    plt.savefig(f"./results/csv/{ANALYZE_TIME}/analyze.png")
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
    plt.savefig(f"./results/csv/{ANALYZE_TIME}/wb_index.png")
    plt.close()


if __name__ == "__main__":
    main()



