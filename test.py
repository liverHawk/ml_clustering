import polars as pl
import argparse
import logging
import coloredlogs

from cluster import Cluster, ClusterConfig

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')
logger = logging.getLogger(__name__)


def load_args():
    parser = argparse.ArgumentParser(description="clustering class run")
    parser.add_argument("-d", "--datasets", type=str, required=True)
    parser.add_argument("-n", "--n_samples", type=int, default=5)
    parser.add_argument("-m", "--method", type=str, default="kmeans")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    args.datasets = args.datasets.split(",")
    return args


def main():
    args = load_args()
    print(args.datasets)

    config = ClusterConfig(
        datasets=args.datasets,
        n_samples=args.n_samples,
        cluster_method=args.method,
        seed=args.seed,
        debug=args.debug
    )

    cluster = Cluster(config)
    cluster.run()


if __name__ == "__main__":
    main()
