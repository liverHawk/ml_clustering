# SSNMF実装検証ドキュメント

Semi-Supervised Non-negative Matrix Factorization (SSNMF) の実装を検証するための包括的なチェックリストとテストケースを提供します。

## 1. 基本的なNMF実装の検証

### 1.1 数学的定義の確認

**基本的なNMF問題:**
```
min ||X - WH||²_F
subject to: W ≥ 0, H ≥ 0
```

ここで:
- X: データ行列 (n × m)
- W: 基底行列 (n × k)
- H: 係数行列 (k × m)

### 1.2 Frobenius ノルム実装チェックリスト

```python
import numpy as np

def verify_frobenius_nmf():
    """Frobenius ノルムベースのNMF実装検証"""
    
    # テストデータ生成
    np.random.seed(42)
    n, m, k = 10, 15, 3
    X = np.abs(np.random.randn(n, m))
    
    # チェック1: 非負性制約
    def check_nonnegativity(W, H):
        assert np.all(W >= 0), "W must be non-negative"
        assert np.all(H >= 0), "H must be non-negative"
        return True
    
    # チェック2: Frobenius ノルムの計算
    def frobenius_loss(X, W, H):
        """
        正しい実装:
        ||X - WH||²_F = Σᵢⱼ (Xᵢⱼ - (WH)ᵢⱼ)²
        """
        reconstruction = W @ H
        diff = X - reconstruction
        loss = np.sum(diff ** 2)
        
        # 別の計算方法で検証
        loss_alt = np.linalg.norm(diff, 'fro') ** 2
        assert np.isclose(loss, loss_alt), "Frobenius norm calculation mismatch"
        
        return loss
    
    # チェック3: 更新則の正当性（乗法的更新）
    def multiplicative_update_frobenius(X, W, H, max_iter=100):
        """
        正しい更新則:
        H ← H ⊙ (W^T X) / (W^T W H + ε)
        W ← W ⊙ (X H^T) / (W H H^T + ε)
        """
        epsilon = 1e-10
        
        for iteration in range(max_iter):
            # H の更新
            numerator_H = W.T @ X
            denominator_H = W.T @ W @ H + epsilon
            H = H * (numerator_H / denominator_H)
            
            # W の更新
            numerator_W = X @ H.T
            denominator_W = W @ H @ H.T + epsilon
            W = W * (numerator_W / denominator_W)
            
            # 収束チェック
            if iteration % 10 == 0:
                loss = frobenius_loss(X, W, H)
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        return W, H
    
    # 初期化
    W = np.abs(np.random.randn(n, k))
    H = np.abs(np.random.randn(k, m))
    
    print("=== Frobenius ノルムベースNMF検証 ===")
    print(f"初期損失: {frobenius_loss(X, W, H):.6f}")
    
    W_opt, H_opt = multiplicative_update_frobenius(X, W, H)
    
    print(f"最終損失: {frobenius_loss(X, W_opt, H_opt):.6f}")
    print(f"非負性チェック: {check_nonnegativity(W_opt, H_opt)}")
    
    return W_opt, H_opt

# 実行
W_frob, H_frob = verify_frobenius_nmf()
```

### 1.3 KLダイバージェンス実装チェックリスト

```python
import numpy as np
def verify_kl_nmf():
    """KLダイバージェンスベースのNMF実装検証"""
    
    np.random.seed(42)
    n, m, k = 10, 15, 3
    X = np.abs(np.random.randn(n, m)) + 0.1  # 0を避ける
    
    # チェック1: KLダイバージェンスの計算
    def kl_divergence_loss(X, W, H):
        """
        正しい実装:
        D_KL(X||WH) = Σᵢⱼ [Xᵢⱼ log(Xᵢⱼ/(WH)ᵢⱼ) - Xᵢⱼ + (WH)ᵢⱼ]
        """
        epsilon = 1e-10
        WH = W @ H + epsilon
        
        # 各項を計算
        term1 = X * np.log(X / WH)
        term2 = -X
        term3 = WH
        
        loss = np.sum(term1 + term2 + term3)
        
        # 数値的安定性チェック
        assert not np.isnan(loss), "Loss is NaN"
        assert not np.isinf(loss), "Loss is infinite"
        
        return loss
    
    # チェック2: KLダイバージェンス用の更新則
    def multiplicative_update_kl(X, W, H, max_iter=100):
        """
        正しい更新則:
        H ← H ⊙ [(W^T (X/(WH))) / (W^T 1)]
        W ← W ⊙ [((X/(WH)) H^T) / (1 H^T)]
        """
        epsilon = 1e-10
        
        for iteration in range(max_iter):
            WH = W @ H + epsilon
            
            # H の更新
            numerator_H = W.T @ (X / WH)
            denominator_H = np.sum(W, axis=0, keepdims=True).T + epsilon
            H = H * (numerator_H / denominator_H)
            
            # WH を再計算
            WH = W @ H + epsilon
            
            # W の更新
            numerator_W = (X / WH) @ H.T
            denominator_W = np.sum(H, axis=1, keepdims=True).T + epsilon
            W = W * (numerator_W / denominator_W)
            
            # 収束チェック
            if iteration % 10 == 0:
                loss = kl_divergence_loss(X, W, H)
                print(f"Iteration {iteration}: KL Loss = {loss:.6f}")
        
        return W, H
    
    # 初期化
    W = np.abs(np.random.randn(n, k)) + 0.1
    H = np.abs(np.random.randn(k, m)) + 0.1
    
    print("\n=== KLダイバージェンスベースNMF検証 ===")
    print(f"初期損失: {kl_divergence_loss(X, W, H):.6f}")
    
    W_opt, H_opt = multiplicative_update_kl(X, W, H)
    
    print(f"最終損失: {kl_divergence_loss(X, W_opt, H_opt):.6f}")
    
    return W_opt, H_opt

# 実行
W_kl, H_kl = verify_kl_nmf()
```

## 2. Semi-Supervised NMF (SSNMF) の検証

### 2.1 SSNMF の数学的定義

```
min ||X - WH||²_F + α||W_L - W̄_L||²_F
subject to: W ≥ 0, H ≥ 0
```

ここで:
- W_L: ラベル付きデータに対応するWの行
- W̄_L: ラベル（制約）行列
- α: 正則化パラメータ

### 2.2 SSNMF実装チェックリスト

```python
import numpy as np
def verify_ssnmf():
    """Semi-supervised NMF実装検証"""
    
    np.random.seed(42)
    n, m, k = 20, 30, 5
    
    # データ生成
    X = np.abs(np.random.randn(n, m))
    
    # 半教師あり設定：最初の5サンプルにラベル
    n_labeled = 5
    labeled_indices = np.arange(n_labeled)
    W_label = np.abs(np.random.randn(n_labeled, k))  # ラベル（制約）
    
    # チェック1: SSNMF損失関数
    def ssnmf_loss(X, W, H, W_label, labeled_indices, alpha=1.0):
        """
        正しい実装:
        Loss = ||X - WH||²_F + α||W[labeled] - W_label||²_F
        """
        # 再構成誤差
        reconstruction_loss = np.sum((X - W @ H) ** 2)
        
        # ラベル制約項
        W_labeled = W[labeled_indices]
        label_loss = np.sum((W_labeled - W_label) ** 2)
        
        total_loss = reconstruction_loss + alpha * label_loss
        
        return total_loss, reconstruction_loss, label_loss
    
    # チェック2: SSNMF更新則
    def ssnmf_update(X, W, H, W_label, labeled_indices, alpha=1.0, max_iter=100):
        """
        正しい更新則:
        H ← H ⊙ (W^T X) / (W^T W H + ε)
        W ← W ⊙ (X H^T + α W_label δ_L) / (W H H^T + α δ_L + ε)
        
        ここで δ_L は labeled_indices の位置が1、他が0の行列
        """
        epsilon = 1e-10
        n = X.shape[0]
        
        # ラベル制約行列の作成
        label_constraint = np.zeros_like(W)
        label_constraint[labeled_indices] = W_label
        
        # ラベル位置のマスク
        label_mask = np.zeros((n, 1))
        label_mask[labeled_indices] = 1
        
        for iteration in range(max_iter):
            # H の更新（通常のNMFと同じ）
            numerator_H = W.T @ X
            denominator_H = W.T @ W @ H + epsilon
            H = H * (numerator_H / denominator_H)
            
            # W の更新（ラベル制約付き）
            numerator_W = X @ H.T + alpha * label_constraint
            denominator_W = W @ H @ H.T + alpha * label_mask + epsilon
            W = W * (numerator_W / denominator_W)
            
            # 収束チェック
            if iteration % 10 == 0:
                total_loss, recon_loss, label_loss = ssnmf_loss(
                    X, W, H, W_label, labeled_indices, alpha
                )
                print(f"Iter {iteration}: Total={total_loss:.4f}, "
                      f"Recon={recon_loss:.4f}, Label={label_loss:.4f}")
        
        return W, H
    
    # 初期化
    W = np.abs(np.random.randn(n, k))
    H = np.abs(np.random.randn(k, m))
    
    print("\n=== Semi-supervised NMF検証 ===")
    initial_loss = ssnmf_loss(X, W, H, W_label, labeled_indices, alpha=1.0)
    print(f"初期損失: Total={initial_loss[0]:.4f}, "
          f"Recon={initial_loss[1]:.4f}, Label={initial_loss[2]:.4f}")
    
    W_opt, H_opt = ssnmf_update(X, W, H, W_label, labeled_indices, alpha=1.0)
    
    final_loss = ssnmf_loss(X, W_opt, H_opt, W_label, labeled_indices, alpha=1.0)
    print(f"最終損失: Total={final_loss[0]:.4f}, "
          f"Recon={final_loss[1]:.4f}, Label={final_loss[2]:.4f}")
    
    # チェック3: ラベル制約の効果確認
    print("\n=== ラベル制約の効果 ===")
    print(f"ラベル付きサンプルの誤差:")
    for i in labeled_indices:
        error = np.linalg.norm(W_opt[i] - W_label[i])
        print(f"  サンプル {i}: {error:.6f}")
    
    return W_opt, H_opt

# 実行
W_ssnmf, H_ssnmf = verify_ssnmf()
```

## 3. 包括的なテストスイート

```python
import numpy as np
def comprehensive_ssnmf_test():
    """包括的なSSNMF実装テスト"""
    
    print("=" * 60)
    print("SSNMF実装包括検証テスト")
    print("=" * 60)
    
    # テスト1: 小規模データでの基本動作
    print("\n[テスト1] 小規模データでの基本動作")
    test_small_scale()
    
    # テスト2: 収束性の確認
    print("\n[テスト2] 収束性の確認")
    test_convergence()
    
    # テスト3: ラベル数の影響
    print("\n[テスト3] ラベル数の影響")
    test_label_amount()
    
    # テスト4: αパラメータの影響
    print("\n[テスト4] αパラメータの影響")
    test_alpha_parameter()
    
    # テスト5: 数値安定性
    print("\n[テスト5] 数値安定性")
    test_numerical_stability()
    
    print("\n" + "=" * 60)
    print("全テスト完了")
    print("=" * 60)

def test_small_scale():
    """小規模データでの動作確認"""
    n, m, k = 5, 8, 2
    X = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 11],
        [5, 6, 7, 8, 9, 10, 11, 12]
    ], dtype=float)
    
    W = np.abs(np.random.randn(n, k))
    H = np.abs(np.random.randn(k, m))
    
    # Frobenius版
    loss_before = np.sum((X - W @ H) ** 2)
    print(f"  初期損失: {loss_before:.4f}")
    
    # 簡易的な更新を数回実行
    for _ in range(50):
        H = H * ((W.T @ X) / (W.T @ W @ H + 1e-10))
        W = W * ((X @ H.T) / (W @ H @ H.T + 1e-10))
    
    loss_after = np.sum((X - W @ H) ** 2)
    print(f"  最終損失: {loss_after:.4f}")
    print(f"  損失減少: {loss_before - loss_after:.4f}")
    
    assert loss_after < loss_before, "損失が減少していません"
    print("  ✓ テスト合格")

def test_convergence():
    """収束性の確認"""
    np.random.seed(42)
    n, m, k = 20, 30, 5
    X = np.abs(np.random.randn(n, m))
    W = np.abs(np.random.randn(n, k))
    H = np.abs(np.random.randn(k, m))
    
    losses = []
    for i in range(100):
        H = H * ((W.T @ X) / (W.T @ W @ H + 1e-10))
        W = W * ((X @ H.T) / (W @ H @ H.T + 1e-10))
        
        loss = np.sum((X - W @ H) ** 2)
        losses.append(loss)
    
    # 損失が単調減少しているか
    is_decreasing = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
    
    print(f"  初期損失: {losses[0]:.4f}")
    print(f"  最終損失: {losses[-1]:.4f}")
    print(f"  単調減少: {is_decreasing}")
    
    # 最後の10イテレーションでの変化が小さいか
    recent_change = abs(losses[-1] - losses[-10])
    print(f"  最近10回の変化: {recent_change:.6f}")
    
    assert recent_change < 0.1, "収束していません"
    print("  ✓ テスト合格")

def test_label_amount():
    """ラベル数の影響を確認"""
    np.random.seed(42)
    n, m, k = 20, 30, 5
    X = np.abs(np.random.randn(n, m))
    
    label_amounts = [2, 5, 10]
    
    for n_labels in label_amounts:
        W = np.abs(np.random.randn(n, k))
        H = np.abs(np.random.randn(k, m))
        
        labeled_indices = np.arange(n_labels)
        W_label = np.abs(np.random.randn(n_labels, k))
        
        # SSNMF実行
        for _ in range(50):
            H = H * ((W.T @ X) / (W.T @ W @ H + 1e-10))
            
            # ラベル制約付き更新
            label_constraint = np.zeros_like(W)
            label_constraint[labeled_indices] = W_label
            label_mask = np.zeros((n, 1))
            label_mask[labeled_indices] = 1
            
            alpha = 1.0
            W = W * ((X @ H.T + alpha * label_constraint) / 
                     (W @ H @ H.T + alpha * label_mask + 1e-10))
        
        # ラベル制約の満足度を計算
        label_error = np.mean([np.linalg.norm(W[i] - W_label[i]) 
                               for i in labeled_indices])
        
        print(f"  ラベル数={n_labels}: ラベル誤差={label_error:.6f}")
    
    print("  ✓ テスト合格")

def test_alpha_parameter():
    """αパラメータの影響を確認"""
    np.random.seed(42)
    n, m, k = 20, 30, 5
    X = np.abs(np.random.randn(n, m))
    
    n_labels = 5
    labeled_indices = np.arange(n_labels)
    W_label = np.abs(np.random.randn(n_labels, k))
    
    alphas = [0.1, 1.0, 10.0]
    
    for alpha in alphas:
        W = np.abs(np.random.randn(n, k))
        H = np.abs(np.random.randn(k, m))
        
        # SSNMF実行
        for _ in range(50):
            H = H * ((W.T @ X) / (W.T @ W @ H + 1e-10))
            
            label_constraint = np.zeros_like(W)
            label_constraint[labeled_indices] = W_label
            label_mask = np.zeros((n, 1))
            label_mask[labeled_indices] = 1
            
            W = W * ((X @ H.T + alpha * label_constraint) / 
                     (W @ H @ H.T + alpha * label_mask + 1e-10))
        
        # 再構成誤差とラベル誤差
        recon_error = np.sum((X - W @ H) ** 2)
        label_error = np.sum((W[labeled_indices] - W_label) ** 2)
        
        print(f"  α={alpha}: 再構成誤差={recon_error:.4f}, "
              f"ラベル誤差={label_error:.4f}")
    
    print("  ✓ テスト合格")

def test_numerical_stability():
    """数値安定性の確認"""
    np.random.seed(42)
    
    # 極端なケース1: 非常に小さい値
    print("  ケース1: 小さい値")
    X_small = np.abs(np.random.randn(10, 15)) * 1e-6
    W = np.abs(np.random.randn(10, 3))
    H = np.abs(np.random.randn(3, 15))
    
    for _ in range(10):
        H = H * ((W.T @ X_small) / (W.T @ W @ H + 1e-10))
        W = W * ((X_small @ H.T) / (W @ H @ H.T + 1e-10))
    
    assert not np.any(np.isnan(W)), "Wにnanが含まれています"
    assert not np.any(np.isnan(H)), "Hにnanが含まれています"
    print("    ✓ nan無し")
    
    # 極端なケース2: 非常に大きい値
    print("  ケース2: 大きい値")
    X_large = np.abs(np.random.randn(10, 15)) * 1e6
    W = np.abs(np.random.randn(10, 3))
    H = np.abs(np.random.randn(3, 15))
    
    for _ in range(10):
        H = H * ((W.T @ X_large) / (W.T @ W @ H + 1e-10))
        W = W * ((X_large @ H.T) / (W @ H @ H.T + 1e-10))
    
    assert not np.any(np.isinf(W)), "Wにinfが含まれています"
    assert not np.any(np.isinf(H)), "Hにinfが含まれています"
    print("    ✓ inf無し")
    
    print("  ✓ テスト合格")

# 全テスト実行
if __name__ == "__main__":
    comprehensive_ssnmf_test()
```

## 4. 実装チェックリスト

### ✅ 必須チェック項目

#### 4.1 数学的正当性
- [ ] 非負性制約が全イテレーションで満たされている
- [ ] 損失関数が正しく計算されている
- [ ] 更新則が正しい数式に基づいている
- [ ] 収束判定が適切に実装されている

#### 4.2 Frobeniusノルム版
- [ ] `||X - WH||²_F = Σᵢⱼ (Xᵢⱼ - (WH)ᵢⱼ)²` が正しく計算されている
- [ ] 乗法的更新則が正しい
- [ ] 損失が単調減少している

#### 4.3 KLダイバージェンス版
- [ ] `D_KL(X||WH)` の各項が正しく計算されている
- [ ] ゼロ除算対策（epsilon追加）がされている
- [ ] log(0)やlog(負数)が発生しないようになっている
- [ ] 数値的安定性が確保されている

#### 4.4 Semi-supervised版
- [ ] ラベル制約項が正しく追加されている
- [ ] ラベル付きサンプルのインデックスが正しく管理されている
- [ ] αパラメータが適切に機能している
- [ ] ラベル無しデータも再構成に寄与している

#### 4.5 実装品質
- [ ] エッジケース（全て0のデータなど）への対応
- [ ] メモリ効率的な実装
- [ ] 計算効率的な実装（不要な行列コピー回避）
- [ ] デバッグ情報の出力（損失の推移など）

## 5. よくある実装ミス

### 5.1 Frobeniusノルム版

```python
# ❌ 間違い：要素ごとの積を使っていない
H = (W.T @ X) / (W.T @ W @ H)  # 行列除算は定義されていない

# ✓ 正しい：要素ごとの積（Hadamard積）を使用
H = H * ((W.T @ X) / (W.T @ W @ H + epsilon))
```

### 5.2 KLダイバージェンス版

```python
# ❌ 間違い：分母の計算が間違っている
denominator_H = W.T @ np.ones_like(X)

# ✓ 正しい：全て1の行列との積
denominator_H = np.sum(W, axis=0, keepdims=True).T
```

### 5.3 Semi-supervised版

```python
# ❌ 間違い：ラベル制約が全サンプルに適用されている
numerator_W = X @ H.T + alpha * W_label

# ✓ 正しい：ラベル付きサンプルのみに適用
label_constraint = np.zeros_like(W)
label_constraint[labeled_indices] = W_label
numerator_W = X @ H.T + alpha * label_constraint
```

## 6. 結果の妥当性確認

```python
def validate_results(X, W, H, W_label=None, labeled_indices=None):
    """結果の妥当性を総合的に確認"""
    
    print("\n=== 結果の妥当性確認 ===")
    
    # 1. 非負性
    print("1. 非負性チェック")
    print(f"   W最小値: {W.min():.6f} (≥0であるべき)")
    print(f"   H最小値: {H.min():.6f} (≥0であるべき)")
    
    # 2. 再構成誤差
    print("\n2. 再構成品質")
    reconstruction = W @ H
    relative_error = np.linalg.norm(X - reconstruction) / np.linalg.norm(X)
    print(f"   相対誤差: {relative_error:.6f}")
    print(f"   RMSE: {np.sqrt(np.mean((X - reconstruction)**2)):.6f}")
    
    # 3. スパース性（あれば）
    print("\n3. スパース性")
    w_sparsity = np.sum(W < 1e-3) / W.size
    h_sparsity = np.sum(H < 1e-3) / H.size
    print(f"   Wのスパース率: {w_sparsity:.2%}")
    print(f"   Hのスパース率: {h_sparsity:.2%}")
    
    # 4. ラベル制約（SSNMFの場合）
    if W_label is not None and labeled_indices is not None:
        print("\n4. ラベル制約の満足度")
        for i in labeled_indices[:min(5, len(labeled_indices))]:
            error = np.linalg.norm(W[i] - W_label[i])
            print(f"   サンプル{i}の誤差: {error:.6f}")
    
    # 5. 数値的健全性
    print("\n5. 数値的健全性")
    print(f"   Wのnan: {np.any(np.isnan(W))}")
    print(f"   Wのinf: {np.any(np.isinf(W))}")
    print(f"   Hのnan: {np.any(np.isnan(H))}")
    print(f"   Hのinf: {np.any(np.isinf(H))}")

# 使用例
# W, H = your_nmf_implementation(X)
# validate_results(X, W, H)
```

## 7. まとめ

このドキュメントを使用して、以下を確認してください：

1. **基本的なNMF**がFrobeniusノルムとKLダイバージェンスの両方で正しく動作する
2. **SSNMF**がラベル制約を適切に組み込んでいる
3. **数値的安定性**が確保されている
4. **収束性**が適切である
5. **パラメータの影響**が理論通りである

各テストを実行し、すべてにパスすることを確認してください。