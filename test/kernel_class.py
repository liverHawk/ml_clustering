import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clustering_methods.configs import GowerSSKNMFConfig
from clustering_methods import GowerSSKNMF
from lib.cluster_index import ClusterIndex

import argparse
import logging
import coloredlogs
import polars as pl
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def load_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--datasets", type=str, default="all")
    parser.add_argument("-d", "--dataset", type=str, default="CICIDS2017_flow_improved")
    parser.add_argument("-v", "--verbose", action='store_true')
    # parser.add_argument("-n", "--n_samples", type=int, default=0)
    args = parser.parse_args()
    # args.datasets = [ds.strip() for ds in args.datasets.split(",")]
    return args


def main():
    time_string = f'{time.strftime("%Y%m%d-%H%M%S")}'
    save_path = Path(f'./results/{time_string}')
    save_path.mkdir(parents=True, exist_ok=True)

    args = load_args()

    config = GowerSSKNMFConfig(
        base_path="/home/toshi/Documents/dataset/project/cleaned",
        dataset_name="CICIDS2017_flow_improved",
        n_samples_per_label=100,
        labeled_rate=0.4,
        random_state=42
    )
    model = GowerSSKNMF(config)
    logger.info(model.get_labels())
    model.set_labels(
        use_labels=['Heartbleed', 'Infiltration', 'Bot', 'DoS', 'DDoS', 'SSH-Patator', 'PortScan', 'FTP-Patator', 'Web Attack'],
        known_labels=["DoS", "FTP-Patator", "Bot"]
    )
    model.set_cols(
        categorical_columns=["Src Port", "Dst Port", "Protocol"]
    )
    model.convert_to_kernel()
    
    score = ClusterIndex(with_label=True)

    start = 9 - 4
    end = 9 + 5
    
    for n_clusters in range(start, end):
        predictions, membership = model.predict(n_clusters, 0.3, 100, 1e-6)

        # logger.info(predictions)
        # logger.info(membership)

        evaluation = pl.DataFrame({
            "kernel": model.kernel,
            "predictions": predictions,
            "true_labels": model.df_setup["Label"].to_list(),
        })

        score.add(
            n_clusters,
            evaluation["kernel"].to_numpy(),
            evaluation["predictions"].to_numpy(),
            evaluation["true_labels"].to_numpy(),
        )

    score.plot(
        path=save_path / "kernel_class_",
        separate_plots=False
    )
    score.save_data(
        path=save_path
    )


    # logger.info(score.get_results())



if __name__ == "__main__":
    main()
