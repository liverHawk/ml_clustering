# SSNMF 使い方ガイド

Semi-Supervised Non-negative Matrix Factorization (SSNMF) ライブラリの包括的な使い方ガイドです。

## 📚 目次

- [基本的な使い方](#基本的な使い方)
  - [1. Frobeniusノルム版（ラベル無し）](#1-frobeniusノルム版ラベル無し)
  - [2. KLダイバージェンス版（ラベル無し）](#2-klダイバージェンス版ラベル無し)
- [ラベル付きデータでの使い方](#ラベル付きデータでの使い方)
  - [3. Semi-supervised NMF（ラベル制約付き）](#3-semi-supervised-nmfラベル制約付き)
  - [4. αパラメータの効果](#4-αパラメータの効果)
- [グラフ正則化版の使い方](#グラフ正則化版の使い方)
  - [5. SSNMFDFrobenius（グラフLaplacian正則化）](#5-ssnmfdfrobeniusグラフlaplacian正則化)
  - [6. グラフ正則化 + ラベル制約](#6-グラフ正則化--ラベル制約)
  - [7. SSNMFDKL（KLダイバージェンス + グラフ正則化）](#7-ssnmfdklklダイバージェンス--グラフ正則化)
- [実践的な例](#実践的な例)
  - [8. クラスタリングへの応用](#8-クラスタリングへの応用)
  - [9. 次元削減への応用](#9-次元削減への応用)
  - [10. NNDSVD初期化の使用](#10-nndsvd初期化の使用)
- [既存コードからの移行](#既存コードからの移行)
  - [11. 既存コードの動作](#11-既存コードの動作)
  - [12. 新しいAPIへの移行](#12-新しいapiへの移行)
- [各クラスの使い分け](#各クラスの使い分け)
- [Tips](#tips)

---

## 基本的な使い方

### 1. Frobeniusノルム版（ラベル無し）

最もシンプルなNMF（教師なし学習）：

```python
import numpy as np
from clustering_methods import SSNMFFrobenius, create_frobenius_config

# データ準備（非負行列）
X = np.abs(np.random.randn(100, 50))  # 100サンプル × 50特徴

# 設定作成
config = create_frobenius_config(
    n_components=10,      # 成分数（k）
    max_iter=100,         # 最大反復回数
    tol=1e-4,            # 収束判定の閾値
    random_state=42,     # 乱数シード
    verbose=True,        # ログ表示
    alpha=0.0            # ラベル制約なし
)

# モデル作成と学習
model = SSNMFFrobenius(config)
model.fit(X)

# 結果取得
W = model.W  # 基底行列 (100 × 10)
H = model.H  # 係数行列 (10 × 50)
X_reconstructed = model.reconstruct()  # 再構成 = W @ H
```

**数学的定義:**
```
min ||X - WH||²_F
subject to: W ≥ 0, H ≥ 0
```

---

### 2. KLダイバージェンス版（ラベル無し）

KLダイバージェンスを使ったNMF：

```python
from clustering_methods import SSNMFKL, create_kl_config

# データ準備（非負、ゼロを避ける）
X = np.abs(np.random.randn(100, 50)) + 0.1

# 設定作成
config = create_kl_config(
    n_components=10,
    max_iter=100,
    verbose=True,
    alpha=0.0  # ラベル制約なし
)

# モデル作成と学習
model = SSNMFKL(config)
model.fit(X)

# 結果取得
W = model.W
H = model.H
```

**数学的定義:**
```
min D_KL(X||WH) = Σᵢⱼ [Xᵢⱼ log(Xᵢⱼ/(WH)ᵢⱼ) - Xᵢⱼ + (WH)ᵢⱼ]
subject to: W ≥ 0, H ≥ 0
```

---

## ラベル付きデータでの使い方

### 3. Semi-supervised NMF（ラベル制約付き）

一部のサンプルにラベル（W行列の制約）を付けて学習：

```python
from clustering_methods import SSNMFFrobenius, create_frobenius_config

# データ準備
X = np.abs(np.random.randn(100, 50))

# ラベル付きサンプルの設定
n_labeled = 10
labeled_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # ラベル付きサンプルのインデックス
W_label = np.abs(np.random.randn(n_labeled, 10))  # ラベル制約行列 (10 × 10)

# 設定作成（alpha > 0でラベル制約を有効化）
config = create_frobenius_config(
    n_components=10,
    alpha=1.0,  # ラベル制約の強度（大きいほど強く制約）
    max_iter=100,
    verbose=True
)

# モデル作成と学習
model = SSNMFFrobenius(config)
model.fit(X, labels=W_label, labeled_indices=labeled_indices)

# 結果確認
W = model.W
print(f"ラベル制約誤差: {np.linalg.norm(W[labeled_indices] - W_label)}")
```

**数学的定義:**
```
min ||X - WH||²_F + α||W_L - W̄_L||²_F
subject to: W ≥ 0, H ≥ 0

ここで:
- W_L: labeled_indicesに対応するWの行
- W̄_L: ラベル制約行列
- α: ラベル制約の強度
```

---

### 4. αパラメータの効果

αを変えることで、再構成精度とラベル制約のバランスを調整できます：

```python
# αを変えて実験
for alpha in [0.1, 1.0, 10.0]:
    config = create_frobenius_config(n_components=10, alpha=alpha, verbose=False)
    model = SSNMFFrobenius(config)
    model.fit(X, labels=W_label, labeled_indices=labeled_indices)

    # ラベル制約誤差と再構成誤差
    label_error = np.linalg.norm(model.W[labeled_indices] - W_label)
    recon_error = np.linalg.norm(X - model.W @ model.H)

    print(f"α={alpha}: ラベル誤差={label_error:.4f}, 再構成誤差={recon_error:.4f}")
```

**出力例:**
```
α=0.1: ラベル誤差=4.6523, 再構成誤差=12.3456
α=1.0: ラベル誤差=2.1234, 再構成誤差=15.6789  # αが大きい→ラベル誤差小、再構成誤差大
α=10.0: ラベル誤差=0.5678, 再構成誤差=20.1234
```

**トレードオフ:**
- α小さい → 再構成を優先、ラベル制約は緩い
- α大きい → ラベル制約を優先、再構成は多少犠牲に

---

## グラフ正則化版の使い方

### 5. SSNMFDFrobenius（グラフLaplacian正則化）

特徴空間でのグラフ構造を考慮したNMF：

```python
from clustering_methods import SSNMFDFrobenius, create_ssnmfd_config

# データ準備
X = np.abs(np.random.randn(100, 50))

# 設定作成
config = create_ssnmfd_config(
    n_components=10,
    alpha=0.0,        # ラベル制約なし
    gamma=0.1,        # グラフ正則化の強度
    n_neighbors=5,    # k-NNグラフの近傍数
    max_iter=100,
    verbose=True
)

# モデル作成と学習
model = SSNMFDFrobenius(config)
model.fit(X)  # グラフは自動的に構築される

# 結果取得
W = model.W
H = model.H
print(f"グラフLaplacian: {model.L.shape}")  # (50 × 50)
```

**数学的定義:**
```
min ||X - WH||²_F + γ Tr(H L H^T)
subject to: W ≥ 0, H ≥ 0

ここで:
- L: グラフLaplacian行列 (L = D - A)
- γ: グラフ正則化の強度
```

**グラフ正則化の効果:**
- 特徴空間で近い位置にある特徴は、同じような係数を持つように正則化
- データのマニフォールド構造を考慮した分解

---

### 6. グラフ正則化 + ラベル制約

両方を組み合わせた使い方：

```python
# ラベル付きデータでグラフ正則化
config = create_ssnmfd_config(
    n_components=10,
    alpha=1.0,      # ラベル制約あり
    gamma=0.1,      # グラフ正則化あり
    n_neighbors=5,
    max_iter=100
)

model = SSNMFDFrobenius(config)
model.fit(X, labels=W_label, labeled_indices=labeled_indices)
```

**数学的定義:**
```
min ||X - WH||²_F + α||W_L - W̄_L||²_F + γ Tr(H L H^T)
subject to: W ≥ 0, H ≥ 0
```

---

### 7. SSNMFDKL（KLダイバージェンス + グラフ正則化）

```python
from clustering_methods import SSNMFDKL, create_ssnmfd_config

X = np.abs(np.random.randn(100, 50)) + 0.1

config = create_ssnmfd_config(
    n_components=10,
    alpha=0.0,
    gamma=0.1,
    n_neighbors=5
)

model = SSNMFDKL(config)
model.fit(X)
```

**数学的定義:**
```
min D_KL(X||WH) + γ Tr(H L H^T)
subject to: W ≥ 0, H ≥ 0
```

---

## 実践的な例

### 8. クラスタリングへの応用

NMFの基底行列Wを使ってクラスタリング：

```python
from clustering_methods import SSNMFFrobenius, create_frobenius_config
import numpy as np

# データ準備（例：文書-単語行列）
n_documents = 200
n_words = 500
X = np.abs(np.random.randn(n_documents, n_words))

# NMFでクラスタリング
config = create_frobenius_config(n_components=5, verbose=True)
model = SSNMFFrobenius(config)
model.fit(X)

# 各文書を最大の係数を持つ成分に割り当て
cluster_assignments = np.argmax(model.W, axis=1)
print(f"クラスタ割り当て: {cluster_assignments}")

# 各クラスタのサンプル数
unique, counts = np.unique(cluster_assignments, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"クラスタ {cluster}: {count}文書")
```

---

### 9. 次元削減への応用

高次元データを低次元に圧縮：

```python
# 高次元データを低次元に圧縮
X_high_dim = np.abs(np.random.randn(1000, 10000))  # 1000サンプル × 10000次元

config = create_frobenius_config(n_components=50, verbose=False)
model = SSNMFFrobenius(config)
model.fit(X_high_dim)

# 圧縮された表現（1000 × 50）
X_compressed = model.W
print(f"元の次元: {X_high_dim.shape}, 圧縮後: {X_compressed.shape}")

# 再構成誤差
X_reconstructed = model.reconstruct()
error = np.linalg.norm(X_high_dim - X_reconstructed) / np.linalg.norm(X_high_dim)
print(f"相対再構成誤差: {error:.4f}")
```

---

### 10. NNDSVD初期化の使用

Non-negative Double SVDによる初期化で収束を改善：

```python
from clustering_methods import SSNMFFrobenius, create_frobenius_config
from clustering_methods.utils.nndsvd import nndsvd_initialization, NNDSVDConfig

# データ準備
X = np.abs(np.random.randn(100, 50))

# モデル作成
config = create_frobenius_config(n_components=10, verbose=False)
model = SSNMFFrobenius(config)

# NNDSVD初期化
init_config = NNDSVDConfig(
    X=X,
    rank=10,
    variant="zero",  # "zero", "average", "random"
    eps=1e-6
)
W0, H0 = nndsvd_initialization(init_config)

# 初期値を設定してから学習
model.W = W0
model.H = H0
model.fit(X)
```

**NNDSVDのバリアント:**
- `"zero"`: ゼロ要素はそのまま（デフォルト）
- `"average"`: ゼロ要素を平均値で置き換え
- `"random"`: ゼロ要素を小さなランダム値で置き換え

---

## 既存コードからの移行

### 11. 既存コードの動作

**既存コードは変更不要**で動作します（非推奨警告が表示されます）：

```python
# 既存の書き方（sample_ssnmf_d.py）
from clustering_methods import ssnmf

base_config = ssnmf.BaseNMFConfig(n_components=5, max_iter=100, tol=1e-4, ...)
config = ssnmf.SSNMFDConfig(ssnmf_config=base_config, lambda_reg=0.1, gamma_reg=0.1, ...)
model = ssnmf.SSNMFD(config)  # DeprecationWarning が表示される
model.fit(X)
```

**警告メッセージ:**
```
DeprecationWarning: SSNMFD from ssnmf.py is deprecated.
Use SSNMFDFrobenius from clustering_methods.ssnmf_refactored instead
for proper label constraint support.
```

---

### 12. 新しいAPIへの移行

推奨される新しい書き方：

```python
# 新しい書き方
from clustering_methods import SSNMFDFrobenius, create_ssnmfd_config

config = create_ssnmfd_config(
    n_components=5,
    max_iter=100,
    tol=1e-4,
    alpha=0.0,      # 明示的にラベル制約パラメータを指定
    gamma=0.1,
    n_neighbors=3,
    random_state=42,
    verbose=True
)
model = SSNMFDFrobenius(config)
model.fit(X)
```

**移行のメリット:**
- ✅ 正しいラベル制約の実装（CHECK_DOC.md準拠）
- ✅ 明確なパラメータ名（`lambda_reg`→`alpha`, `gamma_reg`→`gamma`）
- ✅ 一貫したAPI設計
- ✅ 将来的なサポート保証

---

## 各クラスの使い分け

| クラス | 用途 | ラベル制約 | ダイバージェンス | グラフ正則化 | 推奨シーン |
|--------|------|-----------|----------------|--------------|-----------|
| `SSNMFFrobenius` | 基本的なNMF | ✅ | Frobenius | ❌ | 教師なし・半教師あり学習 |
| `SSNMFKL` | KL版NMF | ✅ | KL | ❌ | 確率分布データ（文書分類など） |
| `SSNMFDFrobenius` | グラフ考慮NMF | ✅ | Frobenius | ✅ | マニフォールド構造があるデータ |
| `SSNMFDKL` | KL+グラフ | ✅ | KL | ✅ | 確率分布データ+マニフォールド |

**選択ガイド:**

1. **基本的な使用**: `SSNMFFrobenius` から始める
2. **確率的データ**: `SSNMFKL` を検討（文書-トピック、画像など）
3. **マニフォールド構造**: `SSNMFDFrobenius` でグラフ正則化を追加
4. **ラベル付きデータがある**: すべてのクラスで `alpha > 0` を設定

---

## Tips

### データの準備

1. **非負性の確保**
   ```python
   # 負の値がある場合
   X = np.abs(X)
   # または
   X = np.maximum(X, 0)
   ```

2. **KL版でのゼロ回避**
   ```python
   # ゼロを小さな値で置き換え
   X = X + 0.1
   # または
   X = np.where(X == 0, 1e-10, X)
   ```

3. **正規化**
   ```python
   # 行ごとに正規化（サンプルごと）
   X = X / (X.sum(axis=1, keepdims=True) + 1e-10)
   # 列ごとに正規化（特徴ごと）
   X = X / (X.sum(axis=0, keepdims=True) + 1e-10)
   ```

---

### パラメータ調整

1. **n_components（成分数）**
   - データの複雑さに応じて調整
   - 小さすぎ → 情報損失
   - 大きすぎ → 過学習、計算コスト増
   - 推奨: クロスバリデーションで決定

2. **alpha（ラベル制約強度）**
   - 0: 完全な教師なし学習
   - 0.1〜1.0: 弱いラベル制約
   - 1.0〜10.0: 中程度のラベル制約
   - 10.0以上: 強いラベル制約

3. **gamma（グラフ正則化強度）**
   - 0: グラフ正則化なし
   - 0.01〜0.1: 弱い正則化
   - 0.1〜1.0: 中程度の正則化
   - 1.0以上: 強い正則化

4. **n_neighbors（k-NNグラフ）**
   - 小さい（3〜5）: ローカルな構造を重視
   - 中程度（10〜20）: バランス
   - 大きい（30以上）: グローバルな構造を重視

---

### 収束の確認

```python
# ログを有効化
config = create_frobenius_config(n_components=10, verbose=True)
model = SSNMFFrobenius(config)
model.fit(X)
```

**出力例:**
```
INFO:clustering_methods.ssnmf_refactored:Iter 001: err=193.751461, rel=inf
INFO:clustering_methods.ssnmf_refactored:Iter 002: err=185.977542, rel=4.01e-02
INFO:clustering_methods.ssnmf_refactored:Iter 003: err=179.756796, rel=3.34e-02
...
INFO:clustering_methods.ssnmf_refactored:Iter 100: err=105.410085, rel=4.26e-04
INFO:clustering_methods.ssnmf_refactored:Converged.
```

**収束していない場合:**
- `max_iter`を増やす
- `tol`を緩める（大きくする）
- 初期化を変える（NNDSVD使用）

---

### 初期化の重要性

```python
# ランダム初期化（デフォルト）
model = SSNMFFrobenius(config)
model.fit(X)

# NNDSVD初期化（推奨）
init_config = NNDSVDConfig(X=X, rank=10, variant="zero")
W0, H0 = nndsvd_initialization(init_config)
model.W = W0
model.H = H0
model.fit(X)
```

**効果:**
- 収束速度の向上
- 局所最適解の回避
- より良い分解品質

---

### 再現性の確保

```python
# random_stateを固定
config = create_frobenius_config(
    n_components=10,
    random_state=42  # 固定値
)

# NumPyのシードも固定
np.random.seed(42)
```

---

## トラブルシューティング

### よくある問題

1. **"X must be non-negative" エラー**
   ```python
   # 解決策
   X = np.abs(X)
   ```

2. **NaNやInfが発生**
   ```python
   # データを確認
   assert not np.any(np.isnan(X))
   assert not np.any(np.isinf(X))

   # ゼロを避ける（KL版）
   X = X + 1e-10
   ```

3. **収束しない**
   ```python
   # max_iterを増やす
   config = create_frobenius_config(max_iter=200)

   # tolを緩める
   config = create_frobenius_config(tol=1e-3)
   ```

4. **メモリ不足**
   ```python
   # n_componentsを減らす
   config = create_frobenius_config(n_components=5)

   # データサイズを減らす
   X_sample = X[:1000, :500]  # サンプリング
   ```

---

## 参考文献

- [CHECK_DOC.md](../CHECK_DOC.md) - 数学的定義と検証方法
- [実装プラン](../.claude/plans/deep-zooming-wirth.md) - アーキテクチャ設計

---

## サポート

問題が発生した場合:
1. 検証スクリプトを実行して動作確認
   ```bash
   uv run python verify_ssnmf_phase1.py
   uv run python verify_ssnmf_phase2.py
   uv run python verify_backward_compatibility.py
   ```

2. GitHubでissueを作成

---

このガイドがSSNMFライブラリの効果的な使用に役立つことを願っています！
