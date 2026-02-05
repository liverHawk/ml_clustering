from polars.lazyframe import LazyFrame
import polars as pl
import logging
from typing import Optional, List
from dataclasses import dataclass

from pathlib import Path

from lib.data import get_schema
from . import cicids2017


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str = "CICIDS2017_improved", debug: bool = False, base_path: str = "/home/hawk/Documents/school/dataset/project/cleaned"):
    path = Path(f"{base_path}") / dataset_name
    files = list(path.glob("*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {path}")
    schema = get_schema(files)
    # 遅延評価でメモリ効率を向上（スキーマを事前に指定して型推論を回避）
    dfs: list[LazyFrame] = [pl.scan_csv(file, schema_overrides=schema) for file in files]
    df = pl.concat(dfs).collect()

    delete_columns = ["Src IP", "Dst IP", "Timestamp", "Source IP", "Destination IP", "SimillarHTTP"]
    delete_columns = [col for col in delete_columns if col in df.columns]

    df: pl.DataFrame = df.drop(delete_columns)

    if debug:
        for col in df.columns:
            logger.info(col)
        value_counts = df["Label"].value_counts()
        value_counts.write_csv("./results/csv/value_counts.csv")

    if dataset_name in ["CICIDS2017_improved", "CICIDS2017_flow_improved", "CSECICIDS2018_improved"]:
        df = cicids2017.relabeled_dataset(df)
    elif dataset_name in ["CICDDoS2019"]:
        # df = cicddos2019.relabeled_dataset(df)
        pass
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n_labels = df["Label"].n_unique()

    return df, { "dataset": dataset_name, "n_clusters": n_labels }


def sampling(df: pl.DataFrame, n_samples: int = 5, seed: int = 42):
    value_count_min: int = df["Label"].value_counts()["count"].min()  # ty:ignore[invalid-assignment]
    if n_samples == 0:
        n_samples = value_count_min
    elif value_count_min < n_samples:
        raise ValueError(f"Least label count is less than n_samples: {value_count_min} < {n_samples}")

    df_sampling = df.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(n=n_samples, seed=seed)
    )
    return df_sampling


@dataclass
class DataLoaderConfig:
    """DataLoaderの設定クラス"""
    dataset_name: str = "CICIDS2017_improved"
    n_samples: int = 0  # 0の場合は最小ラベル数を使用
    seed: int = 42
    debug: bool = False
    category_columns_file: Optional[str] = None  # カテゴリカラムのファイルパス（相対パスまたは絶対パス）


class DataLoader:
    """データセットの読み込みと前処理を行うDataLoaderクラス"""
    
    def __init__(self, config: DataLoaderConfig):
        """
        Args:
            config: DataLoaderConfigオブジェクト
        """
        self.config = config
        self.df_raw: Optional[pl.DataFrame] = None
        self.df_sampled: Optional[pl.DataFrame] = None
        self.metadata: Optional[dict] = None
        self.category_columns: List[str] = []
    
    def load(self) -> tuple[pl.DataFrame, dict]:
        """
        データセットを読み込み、サンプリングを行う
        
        Returns:
            tuple[pl.DataFrame, dict]: (サンプリング済みDataFrame, メタデータ)
        """
        # データセットの読み込み
        self.df_raw, metadata = load_dataset(
            self.config.dataset_name,
            self.config.debug
        )
        
        # サンプリング
        self.df_sampled = sampling(
            self.df_raw,
            self.config.n_samples,
            self.config.seed
        )
        
        # メタデータの更新
        metadata["n_samples"] = len(self.df_sampled)
        
        # カテゴリカラムの読み込み
        self._load_category_columns()
        metadata["category_columns"] = self.category_columns
        
        self.metadata = metadata
        
        return self.df_sampled, metadata
    
    def _load_category_columns(self):
        """カテゴリカラムをファイルから読み込む"""
        if self.config.category_columns_file is None:
            # デフォルトのパスを試す
            default_file = Path(__file__).parent.parent / f"dataset_metadata/{self.config.dataset_name}.txt"
            if default_file.exists():
                category_file = default_file
            else:
                logger.warning(f"カテゴリカラムファイルが見つかりません: {default_file}")
                self.category_columns = []
                return
        else:
            category_file = Path(self.config.category_columns_file)
            if not category_file.is_absolute():
                # 相対パスの場合、プロジェクトルートからのパスとして扱う
                category_file = Path(__file__).parent.parent / category_file
        
        if not category_file.exists():
            logger.warning(f"カテゴリカラムファイルが見つかりません: {category_file}")
            self.category_columns = []
            return
        
        try:
            with open(category_file, "r") as f:
                self.category_columns = [line.strip() for line in f if line.strip()]
            
            # DataFrameに存在するカラムのみを保持
            if self.df_sampled is not None:
                self.category_columns = [
                    col for col in self.category_columns 
                    if col in self.df_sampled.columns
                ]
            elif self.df_raw is not None:
                self.category_columns = [
                    col for col in self.category_columns 
                    if col in self.df_raw.columns
                ]
            
            logger.info(f"カテゴリカラムを読み込みました: {len(self.category_columns)}個")
        except Exception as e:
            logger.error(f"カテゴリカラムファイルの読み込みに失敗しました: {e}")
            self.category_columns = []
    
    def get_data(self) -> pl.DataFrame:
        """
        サンプリング済みのDataFrameを取得
        
        Returns:
            pl.DataFrame: サンプリング済みDataFrame
            
        Raises:
            ValueError: データがまだ読み込まれていない場合
        """
        if self.df_sampled is None:
            raise ValueError("データがまだ読み込まれていません。load()を先に呼び出してください。")
        return self.df_sampled.clone()
    
    def get_metadata(self) -> dict:
        """
        メタデータを取得
        
        Returns:
            dict: メタデータ
            
        Raises:
            ValueError: データがまだ読み込まれていない場合
        """
        if self.metadata is None:
            raise ValueError("データがまだ読み込まれていません。load()を先に呼び出してください。")
        return self.metadata.copy()
    
    def get_category_columns(self) -> List[str]:
        """
        カテゴリカラムのリストを取得
        
        Returns:
            List[str]: カテゴリカラムのリスト
        """
        return self.category_columns.copy()
