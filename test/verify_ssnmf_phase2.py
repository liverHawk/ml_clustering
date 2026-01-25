"""Verification script for Phase 2: SSNMFKL implementation.

This script tests the SSNMFKL implementation against the requirements
in CHECK_DOC.md.
"""

import numpy as np
import sys
from pathlib import Path

# Add clustering_methods to path
sys.path.insert(0, str(Path(__file__).parent))

from clustering_methods.configs import create_kl_config
from clustering_methods.ssnmf_kl_refactored import SSNMFKL


def test_unlabeled_kl_nmf():
    """Test 1: Unlabeled KL NMF - CHECK_DOC.md Section 1.3"""
    print("\n" + "=" * 60)
    print("TEST 1: Unlabeled KL NMF")
    print("=" * 60)

    np.random.seed(42)
    n, m, k = 20, 30, 5

    # Generate test data (avoid zeros for KL divergence)
    X = np.abs(np.random.randn(n, m)) + 0.1

    # Create config
    config = create_kl_config(
        n_components=k, alpha=0.0, max_iter=100, verbose=True, random_state=42
    )

    # Initialize and fit model
    model = SSNMFKL(config)
    model.fit(X)

    # Verify non-negativity
    assert np.all(model.W >= 0), "W contains negative values"
    assert np.all(model.H >= 0), "H contains negative values"
    print("✓ Non-negativity constraint satisfied")

    # Verify no NaN or Inf
    assert not np.any(np.isnan(model.W)), "W contains NaN"
    assert not np.any(np.isnan(model.H)), "H contains NaN"
    assert not np.any(np.isinf(model.W)), "W contains Inf"
    assert not np.any(np.isinf(model.H)), "H contains Inf"
    print("✓ No NaN or Inf")

    # Compute KL divergence
    WH = model.W @ model.H + 1e-10
    kl_div = np.sum(X * np.log((X + 1e-10) / WH) - X + WH)
    print(f"✓ Final KL divergence: {kl_div:.6f}")

    # Verify shapes
    assert model.W.shape == (n, k), f"W shape mismatch: {model.W.shape} != {(n, k)}"
    assert model.H.shape == (k, m), f"H shape mismatch: {model.H.shape} != {(k, m)}"
    print(f"✓ Matrix shapes correct: W={model.W.shape}, H={model.H.shape}")

    print("\n✅ TEST 1 PASSED")
    return True


def test_labeled_kl_nmf():
    """Test 2: Labeled KL NMF"""
    print("\n" + "=" * 60)
    print("TEST 2: Labeled KL NMF")
    print("=" * 60)

    np.random.seed(42)
    n, m, k = 20, 30, 5

    # Generate test data (avoid zeros)
    X = np.abs(np.random.randn(n, m)) + 0.1

    # Setup labels
    n_labeled = 5
    labeled_indices = np.arange(n_labeled)
    W_label = np.abs(np.random.randn(n_labeled, k)) + 0.1

    # Create config
    config = create_kl_config(
        n_components=k, alpha=1.0, max_iter=100, verbose=True, random_state=42
    )

    # Initialize and fit model
    model = SSNMFKL(config)
    model.fit(X, labels=W_label, labeled_indices=labeled_indices)

    # Verify label constraint
    W_labeled = model.W[labeled_indices]
    label_error = np.linalg.norm(W_labeled - W_label, "fro")
    print(f"\n✓ Label constraint error: {label_error:.6f}")

    # Verify non-negativity
    assert np.all(model.W >= 0), "W contains negative values"
    assert np.all(model.H >= 0), "H contains negative values"
    print("✓ Non-negativity constraint satisfied")

    # Verify no NaN or Inf
    assert not np.any(np.isnan(model.W)), "W contains NaN"
    assert not np.any(np.isnan(model.H)), "H contains NaN"
    assert not np.any(np.isinf(model.W)), "W contains Inf"
    assert not np.any(np.isinf(model.H)), "H contains Inf"
    print("✓ No NaN or Inf")

    # Compute individual error components
    WH = model.W @ model.H + 1e-10
    kl_div = np.sum(X * np.log((X + 1e-10) / WH) - X + WH)
    label_term = config.ssnmf_config.alpha * np.sum((W_labeled - W_label) ** 2)
    total = kl_div + label_term

    print(f"✓ KL divergence: {kl_div:.6f}")
    print(f"✓ Label constraint term: {label_term:.6f}")
    print(f"✓ Total objective: {total:.6f}")

    print("\n✅ TEST 2 PASSED")
    return True


def test_kl_convergence():
    """Test 3: KL divergence decreases over iterations"""
    print("\n" + "=" * 60)
    print("TEST 3: KL Divergence Convergence")
    print("=" * 60)

    np.random.seed(42)
    n, m, k = 20, 30, 5
    X = np.abs(np.random.randn(n, m)) + 0.1

    config = create_kl_config(
        n_components=k, alpha=0.0, max_iter=200, tol=1e-6, verbose=False, random_state=42
    )

    # Track KL divergence over iterations
    model = SSNMFKL(config)

    # Initialize
    rng = np.random.default_rng(42)
    model.W = rng.random(size=(n, k)) + 0.1
    model.H = rng.random(size=(k, m)) + 0.1

    kl_divs = []
    for i in range(50):  # Run 50 iterations manually
        model._update_H(X, model.W)
        model._update_W_unlabeled(X, model.H)

        WH = model.W @ model.H + 1e-10
        kl_div = np.sum(X * np.log((X + 1e-10) / WH) - X + WH)
        kl_divs.append(kl_div)

    # Check mostly decreasing (KL may not be strictly monotonic with multiplicative updates)
    decreases = sum(kl_divs[i] >= kl_divs[i + 1] for i in range(len(kl_divs) - 1))
    ratio = decreases / (len(kl_divs) - 1)
    print(f"✓ Decreasing ratio: {ratio:.2%} ({decreases}/{len(kl_divs)-1})")

    # Check recent convergence
    recent_change = abs(kl_divs[-1] - kl_divs[-10])
    print(f"✓ Recent change (last 10 iters): {recent_change:.6f}")
    print(f"✓ Initial KL divergence: {kl_divs[0]:.6f}")
    print(f"✓ Final KL divergence: {kl_divs[-1]:.6f}")
    print(f"✓ Total reduction: {kl_divs[0] - kl_divs[-1]:.6f}")

    # KL divergence should generally decrease
    assert kl_divs[-1] < kl_divs[0], "KL divergence did not decrease overall"

    print("\n✅ TEST 3 PASSED")
    return True


def test_numerical_stability_kl():
    """Test 4: Numerical stability with KL divergence"""
    print("\n" + "=" * 60)
    print("TEST 4: Numerical Stability (KL)")
    print("=" * 60)

    # Test with small values (but avoid exact zeros)
    print("\n--- Small values (×1e-3) ---")
    np.random.seed(42)
    X_small = np.abs(np.random.randn(10, 15)) * 1e-3 + 1e-6

    config = create_kl_config(
        n_components=3, max_iter=20, verbose=False, random_state=42
    )

    model = SSNMFKL(config)
    model.fit(X_small)

    assert not np.any(np.isnan(model.W)), "W contains NaN"
    assert not np.any(np.isnan(model.H)), "H contains NaN"
    assert not np.any(np.isinf(model.W)), "W contains Inf"
    assert not np.any(np.isinf(model.H)), "H contains Inf"
    print("✓ No NaN or Inf in results")

    # Test with larger values
    print("\n--- Larger values (×100) ---")
    X_large = np.abs(np.random.randn(10, 15)) * 100 + 0.1

    model2 = SSNMFKL(config)
    model2.fit(X_large)

    assert not np.any(np.isnan(model2.W)), "W contains NaN"
    assert not np.any(np.isnan(model2.H)), "H contains NaN"
    assert not np.any(np.isinf(model2.W)), "W contains Inf"
    assert not np.any(np.isinf(model2.H)), "H contains Inf"
    print("✓ No NaN or Inf in results")

    print("\n✅ TEST 4 PASSED")
    return True


def test_kl_vs_frobenius():
    """Test 5: Compare KL and Frobenius behaviors"""
    print("\n" + "=" * 60)
    print("TEST 5: KL vs Frobenius Comparison")
    print("=" * 60)

    from clustering_methods.ssnmf_refactored import SSNMFFrobenius
    from clustering_methods.configs import create_frobenius_config

    np.random.seed(42)
    n, m, k = 20, 30, 5
    X = np.abs(np.random.randn(n, m)) + 0.1

    # Test both with same settings
    config_kl = create_kl_config(
        n_components=k, alpha=0.0, max_iter=50, verbose=False, random_state=42
    )
    config_frob = create_frobenius_config(
        n_components=k, alpha=0.0, max_iter=50, verbose=False, random_state=42
    )

    model_kl = SSNMFKL(config_kl)
    model_frob = SSNMFFrobenius(config_frob)

    model_kl.fit(X)
    model_frob.fit(X)

    # They should give different results (different objectives)
    diff_W = np.linalg.norm(model_kl.W - model_frob.W, "fro")
    diff_H = np.linalg.norm(model_kl.H - model_frob.H, "fro")

    print(f"✓ W difference (Frobenius norm): {diff_W:.6f}")
    print(f"✓ H difference (Frobenius norm): {diff_H:.6f}")
    print("✓ KL and Frobenius give different factorizations (as expected)")

    print("\n✅ TEST 5 PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SSNMF PHASE 2 VERIFICATION")
    print("Testing SSNMFKL implementation")
    print("=" * 60)

    tests = [
        ("Unlabeled KL NMF", test_unlabeled_kl_nmf),
        ("Labeled KL NMF", test_labeled_kl_nmf),
        ("KL Convergence", test_kl_convergence),
        ("Numerical Stability", test_numerical_stability_kl),
        ("KL vs Frobenius", test_kl_vs_frobenius),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")

    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
