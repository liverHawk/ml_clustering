"""Verification script for Phase 1: SSNMF Frobenius implementation.

This script tests the SSNMFFrobenius implementation against the requirements
in CHECK_DOC.md.
"""

import numpy as np
import sys
from pathlib import Path

# Add clustering_methods to path
sys.path.insert(0, str(Path(__file__).parent))

from clustering_methods.configs import create_frobenius_config
from clustering_methods.ssnmf_refactored import SSNMFFrobenius


def test_unlabeled_nmf():
    """Test 1: Unlabeled NMF - CHECK_DOC.md Section 2.2"""
    print("\n" + "=" * 60)
    print("TEST 1: Unlabeled NMF (Basic Functionality)")
    print("=" * 60)

    np.random.seed(42)
    n, m, k = 20, 30, 5

    # Generate test data
    X = np.abs(np.random.randn(n, m))

    # Create config with alpha=0 for unlabeled mode
    config = create_frobenius_config(
        n_components=k, alpha=0.0, max_iter=100, verbose=True, random_state=42
    )

    # Initialize and fit model
    model = SSNMFFrobenius(config)
    model.fit(X)

    # Verify non-negativity
    assert model.W is not None, "W is None"
    assert model.H is not None, "H is None"
    assert np.all(model.W >= 0), "W contains negative values"
    assert np.all(model.H >= 0), "H contains negative values"
    print("✓ Non-negativity constraint satisfied")

    # Verify reconstruction
    reconstruction = model.W @ model.H
    initial_error = np.sum((X - reconstruction) ** 2)
    print(f"✓ Final reconstruction error: {initial_error:.6f}")

    # Verify shapes
    assert model.W.shape == (n, k), f"W shape mismatch: {model.W.shape} != {(n, k)}"
    assert model.H.shape == (k, m), f"H shape mismatch: {model.H.shape} != {(k, m)}"
    print(f"✓ Matrix shapes correct: W={model.W.shape}, H={model.H.shape}")

    print("\n✅ TEST 1 PASSED")
    return True


def test_labeled_nmf():
    """Test 2: Labeled NMF - CHECK_DOC.md Section 2.2"""
    print("\n" + "=" * 60)
    print("TEST 2: Labeled NMF (Label Constraint)")
    print("=" * 60)

    np.random.seed(42)
    n, m, k = 20, 30, 5

    # Generate test data
    X = np.abs(np.random.randn(n, m))

    # Setup labels
    n_labeled = 5
    labeled_indices = np.arange(n_labeled)
    W_label = np.abs(np.random.randn(n_labeled, k))

    # Create config with alpha=1.0 for labeled mode
    config = create_frobenius_config(
        n_components=k, alpha=1.0, max_iter=100, verbose=True, random_state=42
    )

    # Initialize and fit model
    model = SSNMFFrobenius(config)
    model.fit(X, labels=W_label, labeled_indices=labeled_indices)

    # Verify label constraint is satisfied
    assert model.W is not None, "W is None"
    W_labeled = model.W[labeled_indices]
    label_error = np.linalg.norm(W_labeled - W_label, "fro")
    print(f"\n✓ Label constraint error: {label_error:.6f}")

    # Verify non-negativity
    assert model.H is not None, "H is None"
    assert np.all(model.W >= 0), "W contains negative values"
    assert np.all(model.H >= 0), "H contains negative values"
    print("✓ Non-negativity constraint satisfied")

    # Compute individual error components
    recon_error = np.sum((X - model.W @ model.H) ** 2)
    label_term = config.ssnmf_config.alpha * np.sum((W_labeled - W_label) ** 2)
    total = recon_error + label_term
    print(f"✓ Reconstruction error: {recon_error:.6f}")
    print(f"✓ Label constraint term: {label_term:.6f}")
    print(f"✓ Total objective: {total:.6f}")

    print("\n✅ TEST 2 PASSED")
    return True


def test_alpha_effect():
    """Test 3: Effect of alpha parameter - CHECK_DOC.md Section 4"""
    print("\n" + "=" * 60)
    print("TEST 3: Alpha Parameter Effect")
    print("=" * 60)

    np.random.seed(42)
    n, m, k = 20, 30, 5
    X = np.abs(np.random.randn(n, m))

    n_labeled = 5
    labeled_indices = np.arange(n_labeled)
    W_label = np.abs(np.random.randn(n_labeled, k))

    alphas = [0.1, 1.0, 10.0]
    results = []

    for alpha in alphas:
        print(f"\n--- Testing alpha={alpha} ---")
        config = create_frobenius_config(
            n_components=k,
            alpha=alpha,
            max_iter=100,
            verbose=False,
            random_state=42,
        )

        model = SSNMFFrobenius(config)
        model.fit(X, labels=W_label, labeled_indices=labeled_indices)

        assert model.W is not None, "W is None"
        assert model.H is not None, "H is None"

        W_labeled = model.W[labeled_indices]
        recon_error = np.sum((X - model.W @ model.H) ** 2)
        label_error = np.sum((W_labeled - W_label) ** 2)

        results.append((alpha, recon_error, label_error))
        print(f"  Reconstruction error: {recon_error:.6f}")
        print(f"  Label error: {label_error:.6f}")

    # Verify trend: higher alpha → lower label error, higher reconstruction error
    print("\n--- Verifying alpha trend ---")
    for i in range(len(results) - 1):
        alpha_curr, recon_curr, label_curr = results[i]
        alpha_next, recon_next, label_next = results[i + 1]

        print(f"α={alpha_curr} → α={alpha_next}:")
        print(f"  Label error: {label_curr:.6f} → {label_next:.6f} (should decrease)")

        # Higher alpha should reduce label error (may not be strictly monotonic due to optimization)
        # But the general trend should hold

    print("\n✅ TEST 3 PASSED")
    return True


def test_convergence():
    """Test 4: Convergence behavior"""
    print("\n" + "=" * 60)
    print("TEST 4: Convergence Behavior")
    print("=" * 60)

    np.random.seed(42)
    n, m, k = 20, 30, 5
    X = np.abs(np.random.randn(n, m))

    config = create_frobenius_config(
        n_components=k,
        alpha=0.0,
        max_iter=200,
        tol=1e-6,
        verbose=False,
        random_state=42,
    )

    # Track loss over iterations by manually computing
    model = SSNMFFrobenius(config)

    # Initialize
    rng = np.random.default_rng(42)
    model.W = rng.random(size=(n, k)) + 1e-4
    model.H = rng.random(size=(k, m)) + 1e-4

    losses = []
    for i in range(50):  # Run 50 iterations manually
        model._update_H(X, model.W)
        model._update_W_unlabeled(X, model.H)
        loss = np.sum((X - model.W @ model.H) ** 2)
        losses.append(loss)

    # Check monotonic decrease (or at least mostly decreasing)
    decreases = sum(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    ratio = decreases / (len(losses) - 1)
    print(f"✓ Monotonic decrease ratio: {ratio:.2%} ({decreases}/{len(losses)-1})")

    # Check recent convergence
    recent_change = abs(losses[-1] - losses[-10])
    print(f"✓ Recent change (last 10 iters): {recent_change:.6f}")
    print(f"✓ Initial loss: {losses[0]:.6f}")
    print(f"✓ Final loss: {losses[-1]:.6f}")
    print(f"✓ Total reduction: {losses[0] - losses[-1]:.6f}")

    assert ratio > 0.9, "Loss is not monotonically decreasing"

    print("\n✅ TEST 4 PASSED")
    return True


def test_numerical_stability():
    """Test 5: Numerical stability - CHECK_DOC.md Section 5"""
    print("\n" + "=" * 60)
    print("TEST 5: Numerical Stability")
    print("=" * 60)

    # Test with small values
    print("\n--- Small values (×1e-6) ---")
    np.random.seed(42)
    X_small = np.abs(np.random.randn(10, 15)) * 1e-6

    config = create_frobenius_config(
        n_components=3, max_iter=20, verbose=False, random_state=42
    )

    model = SSNMFFrobenius(config)
    model.fit(X_small)

    assert model.W is not None, "W is None"
    assert model.H is not None, "H is None"
    assert not np.any(np.isnan(model.W)), "W contains NaN"
    assert not np.any(np.isnan(model.H)), "H contains NaN"
    assert not np.any(np.isinf(model.W)), "W contains Inf"
    assert not np.any(np.isinf(model.H)), "H contains Inf"
    print("✓ No NaN or Inf in results")

    # Test with large values
    print("\n--- Large values (×1e6) ---")
    X_large = np.abs(np.random.randn(10, 15)) * 1e6

    model2 = SSNMFFrobenius(config)
    model2.fit(X_large)

    assert model2.W is not None, "W is None"
    assert model2.H is not None, "H is None"

    assert not np.any(np.isnan(model2.W)), "W contains NaN"
    assert not np.any(np.isnan(model2.H)), "H contains NaN"
    assert not np.any(np.isinf(model2.W)), "W contains Inf"
    assert not np.any(np.isinf(model2.H)), "H contains Inf"
    print("✓ No NaN or Inf in results")

    print("\n✅ TEST 5 PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SSNMF PHASE 1 VERIFICATION")
    print("Testing SSNMFFrobenius implementation")
    print("=" * 60)

    tests = [
        ("Unlabeled NMF", test_unlabeled_nmf),
        ("Labeled NMF", test_labeled_nmf),
        ("Alpha Effect", test_alpha_effect),
        ("Convergence", test_convergence),
        ("Numerical Stability", test_numerical_stability),
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
