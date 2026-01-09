# GowerSSKNMF ハイパーパラメータ一覧

## 変更可能なハイパーパラメータ

### 1. **n_clusters** (クラスタ数)
- **型**: `int`
- **デフォルト**: `len(unique_labels)` (全ラベル数)
- **説明**: クラスタリングするクラスタの数
- **推奨値**: 
  - 既知のクラス数がある場合: その値を使用
  - 不明の場合: エルボー法やシルエット法で決定
  - ネットワーク攻撃: 5〜10クラスタが多い
- **例**: `n_clusters=5`

### 2. **alpha** (ラベル制約の強さ)
- **型**: `float`
- **デフォルト**: `0.1`
- **説明**: 半教師あり学習におけるラベル制約の強さ
- **効果**:
  - `α = 0`: 完全に教師なし学習
  - `α → ∞`: ラベル付きデータに完全に従う
- **推奨値** (ラベル比率に基づく):
  ```python
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
- **調整範囲**: `0.01 ~ 1.0`
- **例**: `alpha=0.1`

### 3. **kernel_method** (カーネル方法)
- **型**: `str`
- **デフォルト**: `'rbf'`
- **選択肢**: `'linear'`, `'rbf'`, `'exponential'`
- **説明**: 距離行列をカーネル行列に変換する方法
- **選択基準**:
  | データ特性 | 推奨method | 理由 |
  |-----------|-----------|------|
  | クラスタが明確に分離 | `linear` | シンプル、速い |
  | クラスタが重複 | `rbf` | 非線形関係を捉える |
  | 不明 | `rbf` | 汎用性が高い |
- **例**: `kernel_method='rbf'`

### 4. **kernel_sigma** (カーネル幅)
- **型**: `float` または `None`
- **デフォルト**: `None` (自動設定: 距離の中央値)
- **説明**: RBF/Exponentialカーネルの幅パラメータ
- **効果**:
  - `σ` 小: 近い点のみ類似 → クラスタが細分化
  - `σ` 大: 遠い点も類似 → クラスタが粗い
- **自動設定**:
  ```python
  dist = model.distance_matrix_
  sigma_auto = np.median(dist[dist > 0])
  ```
- **調整範囲**: `0.1 ~ 2.0`
- **例**: `kernel_sigma=0.5`

### 5. **max_iter** (最大反復回数)
- **型**: `int`
- **デフォルト**: `200`
- **説明**: 最適化の最大反復回数
- **調整指針**:
  - 200で通常十分
  - 収束しない場合は500に増やす
- **例**: `max_iter=200`

### 6. **tol** (収束判定の閾値)
- **型**: `float`
- **デフォルト**: `1e-4`
- **説明**: 収束判定の相対誤差の閾値
- **調整指針**:
  - `1e-4`が標準
  - 精度重視なら`1e-6`
- **例**: `tol=1e-4`

### 7. **random_state** (乱数シード)
- **型**: `int`
- **デフォルト**: `42`
- **説明**: 再現性のための乱数シード
- **例**: `random_state=42`

### 8. **verbose** (進捗表示)
- **型**: `bool`
- **デフォルト**: `False`
- **説明**: 最適化の進捗を表示するかどうか
- **例**: `verbose=True`

## 使用例

```python
model = GowerSSKNMF(
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    n_clusters=9,              # クラスタ数
    kernel_method='rbf',       # カーネル方法
    kernel_sigma=0.5,          # カーネル幅（Noneで自動）
    alpha=0.1,                 # ラベル制約の強さ
    max_iter=200,              # 最大反復回数
    tol=1e-4,                  # 収束判定の閾値
    verbose=True,              # 進捗表示
    random_state=42            # 乱数シード
)
```

## グリッドサーチの例

```python
from sklearn.metrics import adjusted_rand_score

best_score = -np.inf
best_params = None

for alpha in [0.01, 0.05, 0.1, 0.5]:
    for sigma in [0.1, 0.5, 1.0, 2.0]:
        model = GowerSSKNMF(
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            n_clusters=n_clusters,
            kernel_method='rbf',
            kernel_sigma=sigma,
            alpha=alpha,
            verbose=False
        )
        
        cluster_labels = model.fit_predict(
            df_combined.drop("row_index"),
            labeled_indices,
            labels
        )
        
        score = adjusted_rand_score(true_labels, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_params = {'alpha': alpha, 'sigma': sigma}

print(f"Best params: {best_params}")
print(f"Best ARI: {best_score:.4f}")
```
