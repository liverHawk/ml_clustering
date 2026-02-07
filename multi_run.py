from pathlib import Path
import subprocess
import yaml
import logging
import coloredlogs


DATASETS = [
    "CICIDS2017_flow_improved",
    "CSECICIDS2018_improved",
    "CICDDoS2019",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def load_params():
    path = Path(__file__).parent / "params.yaml"
    params = yaml.load(path.read_text(), Loader=yaml.FullLoader)
    return params


def load_labels(dataset: str):
    with open(f"dataset_metadata/{dataset}.txt", "r") as f:
        labels = f.readlines()
    return labels


def main():
    params = load_params()
    for dataset in DATASETS:
        labels = load_labels(dataset)
        
        # 
