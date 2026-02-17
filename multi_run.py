from pathlib import Path
import subprocess
import yaml
import logging
import coloredlogs
import copy


DATASETS = [
    "CICIDS2017_flow_improved",
    "CSECICIDS2018_improved",
    "CICDDoS2019",
]

LABEL_SETS = [
    [
        "DoS GoldenEye",
        "DDoS",
        "PortScan",
    ],
    [
        "DDoS-LOIC-HTTP",
        "SSH-BruteForce",
        "Botnet Ares",
    ],
    [
        "WebDDoS",
        "DrDoS_NetBIOS",
        "DrDoS_SNMP",
    ]
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


def update_params(params: dict, dataset: str, label_set: list) -> dict:
    params['dataset'] = dataset

    known_params = copy.deepcopy(params)
    exclude_params = copy.deepcopy(params)

    known_params['known_labels'] = label_set
    known_params['exclude_labels'] = []

    exclude_params['exclude_labels'] = label_set
    exclude_params['known_labels'] = []

    return known_params, exclude_params


def main():
    params = load_params()
    count = 0
    for dataset, label_set in zip(DATASETS, LABEL_SETS):
        params_sets = update_params(params, dataset, label_set)
        for _params in params_sets:
            count += 1
            with open('params.yaml', 'w') as f:
                yaml.dump(_params, f)
            logger.info(f"[{count:1d}/6] Running {dataset} {label_set}")
            try:
                result = subprocess.run(['dvc', 'repro', '-f'], text=True)
                logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error: {e.stderr}")


if __name__ == '__main__':
    main()

"""
import subprocess
import yaml
import random
import logging
import coloredlogs

from pathlib import Path

coloredlogs.install(level='INFO')
logger = logging.getLogger(__name__)


SEEDS = [22, 42, 62]

DATASETS = [
    'CICIDS2017_flow_improved',
    'CSECICIDS2018_improved',
    'CICDDoS2019',
]

DATASET_INFO = {
    'CICIDS2017_flow_improved': {
        'n_all_labels': 15,
    },
    'CSECICIDS2018_improved': {
        'n_all_labels': 15,
    },
    'CICDDoS2019': {
        'n_all_labels': 19,
    },
}

WAYS = [
    (1, 3),
    (6, 3),
    (10, 3),
    (6, 1),
    (6, 5),
    (1, 1),
    (8, 8),
    (10, 5),
]

"
--- multi_run.py ---

all_labels -> base_class & CIL class

[
CICIDS2017_flow_improved: n_all_labels = 15
CSECICIDS2018_improved: n_all_labels = 15
CICDDoS2019: n_all_labels = 19
]

n_base_class: small, medium, large
n_CIL_class: small, medium, large

(n_base_class, n_CIL_class)
1. (1, 3), (6, 3), (10, 3)
2. (6, 1), (6, 3), (6, 5)
3. (1, 1), (8, 8), (10, 5)

one of base_class is BENIGN
"


def load_params(param_file: Path) -> dict:
    with open(param_file, 'r') as f:
        return yaml.safe_load(f)


def load_all_labels(dataset_name: str) -> list:
    with open(f'dataset_metadata/{dataset_name}.txt', 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() != 'BENIGN']


def pick_up_base_class(all_labels: list, n_base_class: int, seed: int) -> list:
    random.seed(seed)
    base_class = random.sample(all_labels, n_base_class)
    return base_class


def make_combination():
    combinations = {
        'CICIDS2017_flow_improved': {},
        'CSECICIDS2018_improved': {},
        'CICDDoS2019': {},
    }
    for (dataset_name, seed) in zip(DATASETS, SEEDS):
        all_labels = load_all_labels(dataset_name)
        
        # (1, 3), (6, 3), (10, 3)
        for (n_base_class, way) in WAYS:
            pattern = {
                "dataset_name": dataset_name,
                "base_class": n_base_class,
                "num_classes": DATASET_INFO[dataset_name]['n_all_labels'],
                "way": way,
                'seed': seed,
                "create_sessions": {
                    "dataset_name": dataset_name,
                    "base_class": n_base_class,
                    "num_classes": DATASET_INFO[dataset_name]['n_all_labels'],
                    "way": way,
                    'seed': seed,
                    "base_labels": ["BENIGN"] + pick_up_base_class(all_labels, n_base_class - 1, seed),
                }
            }
            combinations[dataset_name][f"{n_base_class}_{way}"] = pattern
    return combinations


def update_params(params: dict, combination: dict) -> dict:
    params['dataset_name'] = combination['dataset_name']
    params['base_class'] = combination['base_class']
    params['num_classes'] = combination['num_classes']
    params['way'] = combination['way']
    params['seed'] = combination['seed']

    params['create_sessions']['dataset_name'] = combination['create_sessions']['dataset_name']
    params['create_sessions']['base_class'] = combination['create_sessions']['base_class']
    params['create_sessions']['num_classes'] = combination['create_sessions']['num_classes']
    params['create_sessions']['way'] = combination['create_sessions']['way']
    params['create_sessions']['seed'] = combination['create_sessions']['seed']
    params['create_sessions']['base_labels'] = combination['create_sessions']['base_labels']

    return params


def main():
    param_file = Path('params.yaml')
    params = load_params(param_file)
    print(params)

    combinations = make_combination()
    print(combinations)

    count = 0
    for dataset_name, combination in combinations.items():
        for combination_name, combination_params in combination.items():
            count += 1
            # dataset_name, combination_name, combination_params
            new_params = update_params(params, combination_params)
            with open('params.yaml', 'w') as f:
                yaml.dump(new_params, f)
            logger.info(f"[{count:2d}/24] Running {dataset_name} {combination_name}")
            try:
                result = subprocess.run(['dvc', 'repro', '-f'], text=True)
                logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error: {e.stderr}")



if __name__ == '__main__':
    main()
"""
