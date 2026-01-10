import numpy as np
from sklearn.neighbors import kneighbors_graph


def build_laplacian(X, n_neighbors):
    knn = kneighbors_graph(
        X.T,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
        metric="euclidean"
    )
    # 疎行列として返ってくるので、対称化後に密行列へ変換してから次数行列を作る
    A = knn.maximum(knn.T).astype(float).toarray()
    D = np.diag(A.sum(axis=1))
    L = D - A
    return A, D, L
