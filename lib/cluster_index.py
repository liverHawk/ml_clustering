import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import polars as pl

from sklearn.cluster import KMeans


def _ensure_dir_prefix(path: str):
    if not path:
        return ""
    if path[-1] not in ["/", "_"]:
        path += "/"

    dir_path = path if path.endswith(os.sep) else os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    return path


def _plot_scores(
        results: dict,
        score_list: list,
        path: str = "",
        separate_plots: bool = False
):
    path = _ensure_dir_prefix(path)
    x_keys = list(results.keys())
    if separate_plots:
        plt.figure(figsize=(10, 5))
        for score in score_list:
            y_values = [
                results[i][score] for i in x_keys
            ]
            plt.plot(x_keys, y_values)
            plt.xlabel('n_clusters')
            plt.ylabel(score.replace("_", " ").title())
            plt.grid(True)
            plt.savefig(f'{path}{score}.png')
            plt.close()
    else:
        # 1枚の図の中に複数のサブプロットを作成 (score_listの数に応じて動的に変更)
        n_scores = len(score_list)
        fig, axes = plt.subplots(n_scores, 1, figsize=(10, 5 * n_scores))
        fig.subplots_adjust(hspace=0.4)  # グラフ間の上下の隙間を調整

        cmap = plt.get_cmap('tab20')
        
        # axesが1次元配列でない場合（指標が1つの場合）の処理
        if n_scores == 1:
            axes = [axes]

        for idx, score in enumerate(score_list):
            y_values = [
                results[i][score] for i in x_keys
            ]
            color = cmap(float(idx) / len(score_list))
            axes[idx].plot(x_keys, y_values, marker='o', color=color)
            axes[idx].set_title(f'{score.replace("_", " ").title()}')
            axes[idx].set_xlabel('n_clusters')
            axes[idx].set_ylabel('Score')
            axes[idx].grid(True)

        plt.savefig(f'{path}clustering_metrics.png')
        plt.close()


class ClusterIndex:
    def __init__(self, with_label=True):
        self.results = {}
        self.with_label = with_label
        if with_label:
            self.results_with_label = {}

    def reset(self):
        self.results = {}
        if self.with_label:
            self.results_with_label = {}

    def get_results(self):
        results = {}
        keys = list(self.results.keys())
        for key in keys:
            results[key] = {
                **self.results[key],
                **(self.results_with_label[key] if self.with_label else {})
            }
        return results

    def __str__(self):
        if self.with_label:
            return str(self.results) + str(self.results_with_label)
        else:
            return str(self.results)

    def _common_add(self, n_clusters, x, y):
        # ラベルが1種類しかない場合、これらの指標は定義できないので NaN を入れてスキップ
        if len(np.unique(y)) < 2:
            self.results[n_clusters] = {
                "silhouette_score": np.nan,
                "ch_score": np.nan,
                "db_score": np.nan
            }
            return

        silhouette_score = metrics.silhouette_score(x, y)
        ch_score = metrics.calinski_harabasz_score(x, y)
        db_score = metrics.davies_bouldin_score(x, y)

        # logger.info(f"n_clusters={n_clusters}, silhouette_score={silhouette_score:.3f}, ch_score={ch_score:.3f}, db_score={db_score:.3f}")

        self.results[n_clusters] = {
            "silhouette_score": silhouette_score,
            "ch_score": ch_score,
            "db_score": db_score
        }

    def _wb_index(self, n_clusters, x, y, kmeans: KMeans):
        overall_center = x.mean(axis=0)
        ssw = kmeans.inertia_
        ssb = sum(
            len(x[y == i]) * np.sum((kmeans.cluster_centers_[i] - overall_center) ** 2) for i in range(kmeans.n_clusters)
        )
        self.results[n_clusters]["wb_index"] = n_clusters * ssw / ssb

    def _cluster_accuracy(self, label, y):
        """
        クラスタリングのaccuracyを計算
        各クラスタに最も多い真のラベルを割り当ててからaccuracyを計算
        """
        # クラスタIDと真のラベルのペアを作成
        cluster_to_label = {}
        for cluster_id in np.unique(y):
            # このクラスタに属するサンプルの真のラベルを取得
            mask = y == cluster_id
            cluster_labels = label[mask]
            # 最も多いラベルをこのクラスタの予測ラベルとする
            # np.uniqueは文字列ラベルにも対応
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            cluster_to_label[cluster_id] = most_common_label
        
        # クラスタラベルを真のラベルにマッピング
        y_mapped = np.array([cluster_to_label[cluster_id] for cluster_id in y])
        
        # accuracyを計算
        accuracy = metrics.accuracy_score(label, y_mapped)
        return accuracy

    def add(self, n_clusters, x, y, label, kmeans: KMeans = None):
        self._common_add(n_clusters, x, y)
        # self._wb_index(n_clusters, x, y, kmeans)

        if not self.with_label:
            return
        # ARI, NMI, FMI
        ari = metrics.adjusted_rand_score(label, y)
        nmi = metrics.normalized_mutual_info_score(label, y)
        fmi = metrics.fowlkes_mallows_score(label, y)
        
        # Accuracy
        accuracy = self._cluster_accuracy(label, y)

        self.results_with_label[n_clusters] = {
            "ARI": ari,
            "NMI": nmi,
            "FMI": fmi,
            "accuracy": accuracy
        }
    
    def get_results(self, n_clusters):
        return {
            **self.results[n_clusters],
            **(self.results_with_label[n_clusters] if self.with_label else {})
        }

    def _normal_plot(self, path="", separate_plots=False):
        if path != "" and path[-1] != "/":
            os.makedirs(
                os.path.dirname(path), exist_ok=True
            )
        _plot_scores(self.results, ["silhouette_score", "ch_score", "db_score"], path, separate_plots)

    def _plot_with_label(self, path="", separate_plots=False):
        if path != "" and path[-1] != "/":
            os.makedirs(
                os.path.dirname(path), exist_ok=True
            )
        _plot_scores(self.results_with_label, ["ARI", "NMI", "FMI", "accuracy"], path, separate_plots)

    def plot(self, path="", separate_plots=False):
        path = str(path)
        self._normal_plot(path, separate_plots)
        if self.with_label:
            self._plot_with_label(path + "label_", separate_plots)
    
    def save_data(self, path=""):
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/results.csv', 'w') as f:
            f.write('n_clusters,' + ','.join(list(self.results[list(self.results.keys())[0]].keys())) + '\n')
            for n_clusters in self.results.keys():
                f.write(f'{n_clusters},' + ','.join(str(self.results[n_clusters][k]) for k in list(self.results[n_clusters].keys())) + '\n')
        with open(f'{path}/results_with_label.csv', 'w') as f:
            f.write('n_clusters,' + ','.join(list(self.results_with_label[list(self.results_with_label.keys())[0]].keys())) + '\n')
            for n_clusters in self.results_with_label.keys():
                f.write(f'{n_clusters},' + ','.join(str(self.results_with_label[n_clusters][k]) for k in list(self.results_with_label[n_clusters].keys())) + '\n')
