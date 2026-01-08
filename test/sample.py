import logging
import coloredlogs

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from lib.cluster_index import ClusterIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# サンプルデータの生成
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=42)

scores = ClusterIndex()

for n_clusters in range(2, 10):
    logger.info(f"n_clusters={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    scores.add(n_clusters, X, cluster_labels)

scores.plot(
    path="./results/sample_",
    separate_plots=False
)
