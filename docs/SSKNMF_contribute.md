# Gower距離ベースSS-KNMF実装ドキュメント

## 目次
1. [概要](#概要)
2. [理論的背景](#理論的背景)
3. [アーキテクチャ](#アーキテクチャ)
4. [実装詳細](#実装詳細)
5. [使用方法](#使用方法)
6. [パラメータ調整ガイド](#パラメータ調整ガイド)
7. [評価方法](#評価方法)
8. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### 目的
混合型データ（カテゴリ変数と数値変数）に対して、少量のラベル付きデータを活用した半教師ありクラスタリングを実行する。

### 特徴
- ✅ **One-Hotエンコーディング不要**: Gower距離でカテゴリを直接扱う
- ✅ **高カーディナリティ対応**: ポート番号などもそのまま使用可能
- ✅ **半教師あり学習**: 少量のラベルで精度向上
- ✅ **スケーラブル**: ベクトル化された実装

### 適用分野
- ネットワーク侵入検知
- 医療診断（検査値 + 症状）
- 顧客セグメンテーション（行動データ + 属性）

---

## 理論的背景

### 1. Gower距離

**定義**:
```
d_Gower(i,j) = Σ δ(i,j,k) * d_k(i,j) / Σ δ(i,j,k)
```

**要素**:
- 数値変数: `d_k = |x_i - x_j| / range_k`
- カテゴリ変数: `d_k = 0 (一致) or 1 (不一致)`

**特性**:
- 範囲: [0, 1]
- 混合型データに対応
- スケール不変

### 2. カーネル行列

**変換方法**:

| 方法 | 式 | 特性 |
|------|-----|------|
| Linear | `K = 1 - D` | シンプル、解釈しやすい |
| RBF | `K = exp(-D²/(2σ²))` | 非線形、滑らか |
| Exponential | `K = exp(-D/σ)` | RBFより鋭敏 |

**選択基準**:
- クラスタが明確に分離: Linear
- クラスタが重複: RBF (σ調整が重要)
- 不明: RBFから開始

### 3. Kernel NMF

**目的関数**:
```
min ||K - HH^T||_F²
```

- K: カーネル行列 (n×n)
- H: 所属度行列 (k×n)
- k: クラスタ数

**更新則** (乗法的更新):
```
H ← H ⊙ √[(KH^T)^T / (HH^TKH^T)^T]
```

### 4. 半教師あり拡張 (SS-KNMF)

**目的関数**:
```
min ||K - HH^T||_F² + α||H_labeled - H_constraint||_F²
```

- α: 制約の強さ (0.01 ~ 1.0)
- H_constraint: ラベル付きサンプルの制約行列

**制約行列**:
```python
H_constraint[label, idx] = 1.0  # ラベル付きサンプル
H_constraint[:, idx].sum() = 1.0  # 確率的制約
```

---

## アーキテクチャ

### クラス構成
```
GowerSSKNMF                    # 高レベルAPI
    ├── gower_distance_vectorized()  # Gower距離計算
    ├── gower_to_kernel()            # カーネル変換
    └── SSKNMF                       # 核アルゴリズム
            ├── fit()                # 学習
            ├── predict()            # 予測
            └── _optimize()          # 最適化ループ
```

### データフロー
```
[入力データ]
    ↓
[Gower距離行列] (n×n)
    ↓
[カーネル行列] (n×n)
    ↓
[SS-KNMF最適化]
    ↓
[H行列] (k×n)
    ↓
[クラスタラベル] (n,)
```

---

## 実装詳細

### 1. Gower距離の計算
```python
def gower_distance_vectorized(df, categorical_cols, numerical_cols):
    """
    ベクトル化されたGower距離計算
    
    計算量: O(n² * f)
    メモリ: O(n²)
    
    最適化:
    - NumPyブロードキャストで高速化
    - ループを最小化
    """
    n = len(df)
    n_features = len(categorical_cols) + len(numerical_cols)
    
    # 数値変数の距離
    num_data = df.select(numerical_cols).to_numpy()
    ranges = np.ptp(num_data, axis=0)  # max - min
    ranges[ranges == 0] = 1  # ゼロ除算回避
    
    # ブロードキャスト: (n,1,f) - (1,n,f) → (n,n,f)
    num_dist = np.sum([
        np.abs(num_data[:, None, k] - num_data[None, :, k]) / ranges[k]
        for k in range(len(numerical_cols))
    ], axis=0)
    
    # カテゴリ変数の距離
    cat_data = df.select(categorical_cols).to_numpy()
    cat_dist = np.sum([
        (cat_data[:, None, k] != cat_data[None, :, k]).astype(float)
        for k in range(len(categorical_cols))
    ], axis=0)
    
    return (num_dist + cat_dist) / n_features
```

**注意点**:
- `ranges[ranges == 0] = 1`: 定数特徴量の処理
- `astype(float)`: bool→floatで加算可能に

### 2. カーネル変換
```python
def gower_to_kernel(distance_matrix, method='linear', sigma=0.5):
    """
    距離→類似度変換
    
    Linear: 最もシンプル、解釈しやすい
    RBF: 非線形関係を捉える、σが重要
    """
    if method == 'linear':
        return 1 - distance_matrix
    
    elif method == 'rbf':
        # σ: カーネルの幅
        # - 小さい: 近い点のみ高類似度（鋭敏）
        # - 大きい: 遠い点も高類似度（滑らか）
        return np.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))
```

**σの選び方**:
```python
# ヒューリスティック: 距離の中央値を使用
sigma = np.median(distance_matrix[distance_matrix > 0])
```

### 3. SS-KNMFの最適化
```python
def _optimize(self, K, H_constraint, labeled_indices):
    """
    乗法的更新則による最適化
    
    収束条件:
    1. 反復回数がmax_iterに達する
    2. 変化量がtol未満
    """
    for iteration in range(self.config.max_iter):
        H_old = self.H_.copy()
        
        # H の更新
        self._update_H(K, H_constraint, unlabeled_mask, labeled_indices)
        
        # 収束判定
        diff = np.linalg.norm(self.H_ - H_old)
        if diff < self.config.tol:
            break
```

**更新式の導出**:

目的関数:
```
L = ||K - HH^T||² + α||H_labeled - H_constraint||²
```

勾配:
```
∂L/∂H = -2(KH^T)^T + 2(HH^TKH^T)^T + 2α(H - H_constraint)
```

乗法的更新則 (KKT条件より):
```
H ← H ⊙ √[(KH^T + αH_constraint)^T / (HH^TKH^T + αH)^T]
```

### 4. 制約の実装
```python
def _update_H(self, K, H_constraint, unlabeled_mask, labeled_indices):
    """
    ラベル付きサンプルは固定、ラベルなしサンプルのみ更新
    """
    # 分子・分母の計算
    numerator = KH_T.T + self.config.alpha * H_constraint
    denominator = HH_TKH_T.T + self.config.alpha * self.H_ + 1e-10
    
    # ラベルなしサンプルのみ更新
    self.H_[:, unlabeled_mask] = self.H_[:, unlabeled_mask] * np.sqrt(
        numerator[:, unlabeled_mask] / denominator[:, unlabeled_mask]
    )
    
    # ラベル付きサンプルは制約で固定
    self.H_[:, labeled_indices] = H_constraint[:, labeled_indices]
```

**なぜ分離するか**:
- ラベル付き: 教師情報を保持
- ラベルなし: データ構造から学習

---

## 使用方法

### 基本的な使い方
```python
import polars as pl
import numpy as np
from gower_ssknmf import GowerSSKNMF

# 1. データの準備
df = pl.read_csv("network_traffic.csv")

# 2. カラムの指定
categorical_cols = ["protocol", "flag", "service"]
numerical_cols = ["duration", "src_bytes", "dst_bytes", "packet_rate"]

# 3. ラベル付きデータ（既知攻撃）
labeled_indices = np.array([0, 100, 200, 300, 400])
labels = np.array([0, 1, 0, 2, 1])  # 0:Normal, 1:DoS, 2:Probe

# 4. モデルの作成
model = GowerSSKNMF(
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    n_clusters=5,
    kernel_method='rbf',
    kernel_sigma=0.5,
    alpha=0.1,
    max_iter=200,
    verbose=True,
    random_state=42
)

# 5. 学習と予測
cluster_labels = model.fit_predict(df, labeled_indices, labels)

# 6. 結果の確認
print("クラスタラベル:", cluster_labels)
print("クラスタごとのサンプル数:", np.bincount(cluster_labels))
```

### 教師なし学習（ラベルなし）
```python
# ラベルを指定しない
cluster_labels = model.fit_predict(df)
```

### 所属確率の取得
```python
# ソフトクラスタリング
membership = model.get_cluster_membership()
print("クラスタ0への所属確率:", membership[0, :])
```

### カスタム距離の使用
```python
# 距離行列を直接指定
from gower_ssknmf import SSKNMF, SSKNMFConfig

# カスタム距離行列
custom_distance = compute_custom_distance(df)
K = 1 - custom_distance

# SS-KNMFに直接入力
config = SSKNMFConfig(n_clusters=5, alpha=0.1)
model = SSKNMF(config)
model.fit(K, labeled_indices, labels)
```

---

## パラメータ調整ガイド

### 1. クラスタ数 (n_clusters)

**選択方法**:
```python
from sklearn.metrics import davies_bouldin_score, silhouette_score

# エルボー法
n_clusters_range = range(2, 11)
dbi_scores = []

for n in n_clusters_range:
    model = GowerSSKNMF(n_clusters=n, ...)
    labels = model.fit_predict(df)
    
    # DBIを計算（カスタム実装が必要）
    dbi = compute_dbi_with_gower(df, labels, model.distance_matrix_)
    dbi_scores.append(dbi)

# エルボーの位置を探す
import matplotlib.pyplot as plt
plt.plot(n_clusters_range, dbi_scores, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.title('Elbow Method')
plt.show()
```

**目安**:
- 既知のクラス数がある場合: その値を使用
- 不明の場合: エルボー法やシルエット法
- ネットワーク攻撃: 5〜10クラスタが多い

### 2. 制約の強さ (alpha)

**効果**:
- `α = 0`: 完全に教師なし
- `α → ∞`: ラベル付きデータに完全に従う

**推奨値**:
```python
# ラベル比率に基づく設定
label_ratio = len(labeled_indices) / len(df)

if label_ratio < 0.01:      # <1%
    alpha = 0.01
elif label_ratio < 0.05:    # 1-5%
    alpha = 0.05
elif label_ratio < 0.1:     # 5-10%
    alpha = 0.1
else:                        # >10%
    alpha = 0.5
```

**グリッドサーチ**:
```python
alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
best_alpha = None
best_score = -np.inf

for alpha in alphas:
    model = GowerSSKNMF(alpha=alpha, ...)
    labels = model.fit_predict(df, labeled_indices, known_labels)
    
    # ARIで評価（真のラベルがある場合）
    score = adjusted_rand_score(true_labels, labels)
    if score > best_score:
        best_score = score
        best_alpha = alpha
```

### 3. カーネルパラメータ (kernel_method, kernel_sigma)

**method の選択**:

| データ特性 | 推奨method | 理由 |
|-----------|-----------|------|
| クラスタが明確に分離 | `linear` | シンプル、速い |
| クラスタが重複 | `rbf` | 非線形関係を捉える |
| 不明 | `rbf` | 汎用性が高い |

**sigma の調整**:
```python
# 自動設定（中央値）
dist = model.distance_matrix_
sigma_auto = np.median(dist[dist > 0])

# 手動調整
sigmas = [0.1, 0.5, 1.0, 2.0]
for sigma in sigmas:
    K = gower_to_kernel(dist, method='rbf', sigma=sigma)
    # 評価...
```

**効果**:
- `σ` 小: 近い点のみ類似 → クラスタが細分化
- `σ` 大: 遠い点も類似 → クラスタが粗い

### 4. 収束パラメータ
```python
config = SSKNMFConfig(
    max_iter=200,      # 最大反復回数
    tol=1e-4,          # 収束判定の閾値
    verbose=True       # 進捗表示
)
```

**調整指針**:
- `max_iter`: 200で通常十分、収束しない場合は500
- `tol`: 1e-4が標準、精度重視なら1e-6

---

## 評価方法

### 1. 内部評価（真のラベルなし）

#### Davies-Bouldin Index (DBI)

**カスタム実装** (Gower距離版):
```python
def compute_dbi_with_gower(df, labels, distance_matrix):
    """
    Gower距離を用いたDBI計算
    
    DBI = (1/k) Σ max_{j≠i} (S_i + S_j) / M_{ij}
    
    S_i: クラスタiの凝集度（平均クラスタ内距離）
    M_{ij}: クラスタi,j間の距離（セントロイド間距離）
    """
    n_clusters = len(np.unique(labels))
    
    # 各クラスタの凝集度
    S = np.zeros(n_clusters)
    for i in range(n_clusters):
        mask = labels == i
        cluster_dists = distance_matrix[np.ix_(mask, mask)]
        if cluster_dists.size > 1:
            S[i] = cluster_dists.mean()
    
    # クラスタ間距離
    M = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            mask_i = labels == i
            mask_j = labels == j
            inter_dists = distance_matrix[np.ix_(mask_i, mask_j)]
            M[i, j] = M[j, i] = inter_dists.mean()
    
    # DBI計算
    dbi = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j and M[i, j] > 0:
                ratio = (S[i] + S[j]) / M[i, j]
                max_ratio = max(max_ratio, ratio)
        dbi += max_ratio
    
    return dbi / n_clusters
```

**解釈**:
- 小さいほど良い (クラスタが凝集し、分離している)
- 絶対値より相対比較が重要

#### Silhouette係数
```python
from sklearn.metrics import silhouette_score

def compute_silhouette_with_gower(labels, distance_matrix):
    """
    Gower距離を用いたSilhouette係数
    """
    return silhouette_score(distance_matrix, labels, metric='precomputed')
```

**解釈**:
- 範囲: [-1, 1]
- 1に近い: 良いクラスタリング
- 0付近: クラスタが重複
- 負: 誤分類の可能性

### 2. 外部評価（真のラベルあり）
```python
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score
)

# Adjusted Rand Index (ARI)
ari = adjusted_rand_score(true_labels, predicted_labels)
# 範囲: [-1, 1]、1が完全一致、0がランダム

# Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)
# 範囲: [0, 1]、1が完全一致

# Fowlkes-Mallows Index
fmi = fowlkes_mallows_score(true_labels, predicted_labels)
# 範囲: [0, 1]、1が完全一致
```

### 3. 統合評価
```python
def evaluate_clustering(df, model, predicted_labels, true_labels=None):
    """
    包括的な評価
    """
    results = {}
    
    # 内部評価
    results['dbi'] = compute_dbi_with_gower(
        df, predicted_labels, model.distance_matrix_
    )
    results['silhouette'] = compute_silhouette_with_gower(
        predicted_labels, model.distance_matrix_
    )
    
    # 外部評価（真のラベルがある場合）
    if true_labels is not None:
        results['ari'] = adjusted_rand_score(true_labels, predicted_labels)
        results['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
    
    # クラスタサイズ
    unique, counts = np.unique(predicted_labels, return_counts=True)
    results['cluster_sizes'] = dict(zip(unique, counts))
    
    # 再構成誤差
    results['reconstruction_error'] = model.get_reconstruction_errors()[-1]
    
    return results

# 使用例
results = evaluate_clustering(df, model, cluster_labels, true_labels)
print("評価結果:")
for key, value in results.items():
    print(f"  {key}: {value}")
```

---

## トラブルシューティング

### 問題1: 収束しない

**症状**:
```
Iteration 199: Reconstruction Error = 5.234, Change = 0.532
最大反復回数に達しました
```

**原因と対策**:

1. **max_iterが不足**
```python
   # 対策: 反復回数を増やす
   config = SSKNMFConfig(max_iter=500)
```

2. **alphaが大きすぎる**
```python
   # 対策: alphaを小さくする
   model = GowerSSKNMF(alpha=0.01)  # 0.1 → 0.01
```

3. **カーネル行列が不適切**
```python
   # 対策: カーネル方法を変える
   model = GowerSSKNMF(kernel_method='linear')  # rbf → linear
```

### 問題2: すべてが1つのクラスタに

**症状**:
```
クラスタごとのサンプル数: [1000, 0, 0, 0, 0]
```

**原因と対策**:

1. **alphaが大きすぎる & ラベルが偏っている**
```python
   # ラベル付きデータが1クラスに偏っている
   labeled_indices = [0, 1, 2]  # すべてクラスタ0
   labels = [0, 0, 0]
   
   # 対策: alphaを小さく、またはラベルを多様化
   model = GowerSSKNMF(alpha=0.01)
```

2. **カーネルのσが大きすぎる**
```python
   # すべてのサンプルが類似→1クラスタに
   # 対策: σを小さくする
   model = GowerSSKNMF(kernel_sigma=0.1)  # 0.5 → 0.1
```

3. **初期化が悪い**
```python
   # 対策: 乱数シードを変える
   model = GowerSSKNMF(random_state=123)  # 42 → 123
```

### 問題3: メモリエラー

**症状**:
```
MemoryError: Unable to allocate array with shape (100000, 100000)
```

**原因**: 距離行列が O(n²) のメモリを消費

**対策**:

1. **サンプリング**
```python
   # データを削減
   df_sample = df.sample(n=10000, random_state=42)
   model.fit_predict(df_sample, ...)
```

2. **ミニバッチ処理**
```python
   def clustering_with_batches(df, batch_size=5000):
       # 代表サンプルを選択
       representatives = df.sample(n=batch_size)
       
       # 代表サンプルでクラスタリング
       model = GowerSSKNMF(...)
       model.fit(representatives, ...)
       
       # 残りを最近傍で割り当て
       # (実装省略)
```

3. **スパース行列の利用**
```python
   from scipy.sparse import csr_matrix
   
   # 閾値以下の類似度を0に
   K[K < 0.01] = 0
   K_sparse = csr_matrix(K)
```

### 問題4: DBIが最小値で真のクラスタ数と一致しない

**症状**:
```
真のクラスタ数: 5
DBIが最小: n_clusters=8
```

**これは正常**: DBIは必ずしも真のクラスタ数で最小にならない

**対策**: エルボー法を使う
```python
import matplotlib.pyplot as plt

n_range = range(2, 11)
dbi_scores = []

for n in n_range:
    model = GowerSSKNMF(n_clusters=n, ...)
    labels = model.fit_predict(df)
    dbi = compute_dbi_with_gower(df, labels, model.distance_matrix_)
    dbi_scores.append(dbi)

# エルボーを探す
plt.plot(n_range, dbi_scores, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('DBI')
plt.title('Elbow Method')

# 勾配の変化を計算
gradients = np.diff(dbi_scores)
elbow = np.argmax(gradients) + 2  # エルボーの位置
print(f"推奨クラスタ数: {elbow}")
```

### 問題5: カテゴリ変数が考慮されていない気がする

**検証方法**:
```python
# カテゴリのみでクラスタリング
model_cat_only = GowerSSKNMF(
    categorical_cols=["protocol", "flag"],
    numerical_cols=[],  # 空
    n_clusters=3
)
labels_cat = model_cat_only.fit_predict(df)

# 数値のみでクラスタリング
model_num_only = GowerSSKNMF(
    categorical_cols=[],  # 空
    numerical_cols=["duration", "src_bytes"],
    n_clusters=3
)
labels_num = model_num_only.fit_predict(df)

# 両方でクラスタリング
model_both = GowerSSKNMF(
    categorical_cols=["protocol", "flag"],
    numerical_cols=["duration", "src_bytes"],
    n_clusters=3
)
labels_both = model_both.fit_predict(df)

# 比較
print("カテゴリのみとの一致度:", adjusted_rand_score(labels_cat, labels_both))
print("数値のみとの一致度:", adjusted_rand_score(labels_num, labels_both))
```

**期待される結果**: 
- 両方の一致度が中程度 (0.3〜0.7) → カテゴリも数値も考慮されている
- 片方が1.0に近い → もう片方が無視されている可能性

---

## よくある質問 (FAQ)

### Q1: One-Hotエンコーディングと比べてどのくらい次元が削減される?

**A**: 例を見てみましょう。
```python
# カテゴリ変数
categorical_cols = ["protocol", "flag", "service"]
# ユニーク値数: 5, 10, 20

# One-Hotの場合
n_features_onehot = 5 + 10 + 20 = 35次元

# Gower距離の場合
n_features_gower = 3次元（カテゴリ数）

# 削減率: (35-3)/35 = 91.4%
```

ポート番号（65535種類）を含む場合、削減効果はさらに大きい。

### Q2: どのくらいのラベル付きデータが必要?

**A**: 一般的な目安:

| ラベル比率 | 効果 | 推奨alpha |
|-----------|------|----------|
| < 0.1% | ほぼ教師なし | 0.001 |
| 0.1-1% | わずかに改善 | 0.01 |
| 1-5% | 明確な改善 | 0.05-0.1 |
| 5-10% | 大きな改善 | 0.1-0.5 |
| > 10% | 教師ありに近い | 0.5-1.0 |

**最低限**: 各クラスタに1サンプル以上

### Q3: 計算時間はどのくらい?

**A**: 主要なボトルネック:
```python
# Gower距離: O(n² * f)
# カーネル変換: O(n²)
# SS-KNMF: O(n² * k * T)
# 合計: O(n² * (f + kT))

# 例: n=10,000, f=10, k=5, T=200
# 時間: 約1-2分（CPU）
```

大規模データ（n > 50,000）では、サンプリングやミニバッチを推奨。

### Q4: scikit-learnのNMFとの違いは?

**A**: 

| 項目 | scikit-learn NMF | SS-KNMF |
|------|-----------------|---------|
| 入力 | 特徴量行列 X | カーネル行列 K |
| 分解 | X ≈ WH | K ≈ HH^T |
| カテゴリ | One-Hot必須 | Gower距離でOK |
| 半教師あり | なし | あり |
| 用途 | トピックモデル等 | クラスタリング |

### Q5: 実装の正しさをどう確認する?

**A**: テストケース:
```python
# テスト1: 明確に分離されたデータ
df_test = pl.DataFrame({
    "x": [1,1,1,10,10,10],
    "y": [1,1,1,10,10,10],
    "cat": ["A","A","A","B","B","B"]
})

model = GowerSSKNMF(
    categorical_cols=["cat"],
    numerical_cols=["x", "y"],
    n_clusters=2
)
labels = model.fit_predict(df_test)

# 期待: [0,0,0,1,1,1] または [1,1,1,0,0,0]
assert len(np.unique(labels)) == 2
assert np.all((labels[:3] == labels[0]) & (labels[3:] == labels[3]))

print("✓ テスト合格")
```

---

## 付録A: 完全なサンプルコード

### CICIDS2017データでの例
```python
import polars as pl
import numpy as np
from gower_ssknmf import GowerSSKNMF
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score

# 1. データ読み込み
df = pl.read_csv("CICIDS2017_sample.csv")

# 2. 特徴量の選択
categorical_cols = [
    "protocol_type",
    "flag",
    "service"
]

numerical_cols = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent"
]

# 3. 既知攻撃のサンプリング（5%）
known_attacks = df.filter(pl.col("label") != "BENIGN")
n_labeled = int(len(known_attacks) * 0.05)
labeled_sample = known_attacks.sample(n=n_labeled, random_state=42)

# ラベルをエンコード
label_map = {label: i for i, label in enumerate(df["label"].unique())}
labeled_indices = labeled_sample.select(pl.col("index")).to_numpy().flatten()
labels = labeled_sample.select(pl.col("label").map_dict(label_map)).to_numpy().flatten()

# 4. グリッドサーチ
best_score = -np.inf
best_params = None

for alpha in [0.01, 0.05, 0.1]:
    for sigma in [0.1, 0.5, 1.0]:
        model = GowerSSKNMF(
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            n_clusters=len(label_map),
            kernel_method='rbf',
            kernel_sigma=sigma,
            alpha=alpha,
            max_iter=200,
            verbose=False,
            random_state=42
        )
        
        predicted = model.fit_predict(df, labeled_indices, labels)
        true_labels = df["label"].map_dict(label_map).to_numpy()
        
        ari = adjusted_rand_score(true_labels, predicted)
        
        if ari > best_score:
            best_score = ari
            best_params = {'alpha': alpha, 'sigma': sigma}
        
        print(f"alpha={alpha}, sigma={sigma}: ARI={ari:.4f}")

print(f"\nBest params: {best_params}")
print(f"Best ARI: {best_score:.4f}")

# 5. 最良パラメータで再学習
final_model = GowerSSKNMF(
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    n_clusters=len(label_map),
    kernel_method='rbf',
    kernel_sigma=best_params['sigma'],
    alpha=best_params['alpha'],
    max_iter=200,
    verbose=True,
    random_state=42
)

final_labels = final_model.fit_predict(df, labeled_indices, labels)

# 6. 評価
print("\n最終評価:")
print(f"  ARI: {adjusted_rand_score(true_labels, final_labels):.4f}")
print(f"  NMI: {normalized_mutual_info_score(true_labels, final_labels):.4f}")
print(f"  DBI: {compute_dbi_with_gower(df, final_labels, final_model.distance_matrix_):.4f}")

# 7. 結果の可視化
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# H行列をPCAで可視化
H = final_model.get_cluster_membership()
pca = PCA(n_components=2)
H_2d = pca.fit_transform(H.T)

plt.figure(figsize=(10, 6))
for i in range(len(label_map)):
    mask = final_labels == i
    plt.scatter(H_2d[mask, 0], H_2d[mask, 1], label=f'Cluster {i}', alpha=0.6)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Cluster Visualization (PCA of H matrix)')
plt.legend()
plt.savefig('cluster_visualization.png')
plt.show()

# 8. 混同行列
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(true_labels, final_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Cluster')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
```

---

## 付録B: パフォーマンス最適化

### 大規模データへの対応
```python
class GowerSSKNMFOptimized(GowerSSKNMF):
    """
    メモリ効率化版
    """
    
    def fit(self, df, labeled_indices=None, labels=None, 
            use_sampling=True, sample_size=10000):
        """
        サンプリングベースの学習
        """
        if use_sampling and len(df) > sample_size:
            # 代表サンプルを選択
            df_sample = df.sample(n=sample_size, random_state=42)
            
            # ラベル付きサンプルを必ず含める
            if labeled_indices is not None:
                labeled_df = df[labeled_indices]
                df_sample = pl.concat([df_sample, labeled_df]).unique()
            
            # サンプルでモデル学習
            super().fit(df_sample, labeled_indices, labels)
            
            # 全データへの割り当て（最近傍）
            self._assign_remaining(df, df_sample)
        else:
            super().fit(df, labeled_indices, labels)
    
    def _assign_remaining(self, df_full, df_sample):
        """
        残りのサンプルを最近傍クラスタに割り当て
        """
        # 実装省略（Gower距離で最近傍探索）
        pass
```

### 並列化
```python
from joblib import Parallel, delayed

def parallel_gower_distance(df, categorical_cols, numerical_cols, n_jobs=-1):
    """
    並列化されたGower距離計算
    """
    n = len(df)
    
    def compute_chunk(i_start, i_end):
        # チャンクごとに距離計算
        chunk_dist = np.zeros((i_end - i_start, n))
        # 計算ロジック...
        return chunk_dist
    
    # 並列実行
    chunk_size = n // (n_jobs if n_jobs > 0 else 4)
    chunks = [(i, min(i+chunk_size, n)) for i in range(0, n, chunk_size)]
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_chunk)(start, end) for start, end in chunks
    )
    
    return np.vstack(results)
```

---

## 付録C: 参考文献

### 理論的基礎

1. **Gower距離**
   - Gower, J. C. (1971). "A general coefficient of similarity and some of its properties". Biometrics, 27(4), 857-871.

2. **NMF**
   - Lee, D. D., & Seung, H. S. (1999). "Learning the parts of objects by non-negative matrix factorization". Nature, 401(6755), 788-791.

3. **Kernel NMF**
   - Zhang, D., Zhou, Z. H., & Chen, S. (2006). "Non-negative matrix factorization on kernels". PRICAI 2006: Trends in Artificial Intelligence, 404-412.

4. **Semi-supervised NMF**
   - Liu, H., Wu, Z., Li, X., Cai, D., & Huang, T. S. (2012). "Constrained nonnegative matrix factorization for image representation". IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(7), 1299-1311.

### 応用例

5. **ネットワーク侵入検知**
   - Buczak, A. L., & Guven, E. (2016). "A survey of data mining and machine learning methods for cyber security intrusion detection". IEEE Communications Surveys & Tutorials, 18(2), 1153-1176.

6. **混合型データクラスタリング**
   - Ahmad, A., & Dey, L. (2007). "A k-mean clustering algorithm for mixed numeric and categorical data". Data & Knowledge Engineering, 63(2), 503-527.

---

## まとめ

このドキュメントでは、Gower距離ベースのSS-KNMFの実装について、理論から実装、使用方法、トラブルシューティングまで包括的に説明しました。

**キーポイント**:
1. One-Hotエンコーディング不要
2. 少量のラベルで精度向上
3. パラメータ調整が重要
4. 評価は複数の指標で

**次のステップ**:
- 実際のデータで実験
- パラメータチューニング
- 論文執筆のための結果整理

質問があれば、お気軽にお問い合わせください。