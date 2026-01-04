import numpy as np


def create_centers_with_distances(n_clusters, n_features, distances, random_state=42):
    """
    指定した距離のクラスタ中心を生成

    Parameters:
    -----------
    n_clusters : int
        クラスタ数
    n_features : int
        次元数
    distances : list or float
        中心間の距離。
        - float: すべての中心間距離をこの値に設定
        - list: 中心ペアごとの距離を指定（対角線要素も含む）
    random_state : int
        乱数シード

    Returns:
    --------
    centers : np.ndarray
        生成された中心座標 (n_clusters, n_features)
    """
    np.random.seed(random_state)

    # 最初の中心をランダムに生成
    centers = np.random.randn(n_clusters, n_features)
    centers[0] = 0  # 最初の中心を原点に固定

    # 距離が指定されている場合は調整
    if isinstance(distances, (int, float)):
        # すべてのペアで同じ距離
        target_dist = distances
        for i in range(1, n_clusters):
            # ランダムな方向で、指定距離だけ離す
            direction = np.random.randn(n_features)
            direction /= np.linalg.norm(direction)
            centers[i] = centers[0] + direction * target_dist

    return centers