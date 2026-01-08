"""
DataLoaderの使用例
"""
import logging
from dataset import DataLoader, DataLoaderConfig
from lib.data import encode_categorical, normalize, EncodeMethod, NormalizeMethod
from typing import get_args
from itertools import product

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """基本的な使用方法の例"""
    # DataLoaderConfigを作成
    config = DataLoaderConfig(
        dataset_name="CICIDS2017_improved",
        n_samples=100,  # 各ラベルから100サンプル取得（0の場合は最小ラベル数を使用）
        seed=42,
        debug=False
    )
    
    # DataLoaderを作成してデータを読み込む
    loader = DataLoader(config)
    df, metadata = loader.load()
    
    logger.info(f"データセット: {metadata['dataset']}")
    logger.info(f"クラスタ数: {metadata['n_clusters']}")
    logger.info(f"サンプル数: {metadata['n_samples']}")
    logger.info(f"カテゴリカラム: {loader.get_category_columns()}")
    
    return df, metadata


def example_with_preprocessing():
    """前処理を含む使用例"""
    # DataLoaderConfigを作成
    config = DataLoaderConfig(
        dataset_name="CICIDS2017_improved",
        n_samples=100,
        seed=42
    )
    
    # DataLoaderを作成してデータを読み込む
    loader = DataLoader(config)
    df, metadata = loader.load()
    
    # エンコーディングと正規化の組み合わせを試す
    encode_methods = get_args(EncodeMethod)
    normalize_methods = get_args(NormalizeMethod)
    
    for encode_method, normalize_method in product(encode_methods, normalize_methods):
        logger.info(f"処理中: encode={encode_method}, normalize={normalize_method}")
        
        # データのコピーを作成
        df_copy = loader.get_data()
        
        # エンコーディング
        category_columns = loader.get_category_columns()
        df_encoded = encode_categorical(df_copy, category_columns, method=encode_method)
        
        # 正規化
        df_normalized = normalize(df_encoded, category_columns, method=normalize_method)
        
        # ここでクラスタリングなどの処理を行う
        # x = df_normalized.drop("Label").to_numpy()
        # y = df_normalized["Label"].to_numpy()
        
        logger.info(f"  列数: {len(df_normalized.columns)}")


if __name__ == "__main__":
    logger.info("=== 基本的な使用例 ===")
    df, metadata = example_basic_usage()
    
    logger.info("\n=== 前処理を含む使用例 ===")
    example_with_preprocessing()

