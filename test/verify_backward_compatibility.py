"""Verification script for backward compatibility.

This script verifies that the old API (used by sample_ssnmf_d.py) still works
with the new implementation structure.
"""

import numpy as np
import sys
from pathlib import Path
import warnings

# Add clustering_methods to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("BACKWARD COMPATIBILITY VERIFICATION")
print("=" * 60)

# Test 1: Check that old imports work
print("\n[TEST 1] Old import pattern (as used in sample_ssnmf_d.py)")
try:
    from clustering_methods import ssnmf
    print("✓ Import successful: from clustering_methods import ssnmf")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check that old classes are accessible
print("\n[TEST 2] Accessing old classes through ssnmf namespace")
try:
    assert hasattr(ssnmf, 'BaseNMFConfig'), "ssnmf.BaseNMFConfig not found"
    print("✓ ssnmf.BaseNMFConfig accessible")

    assert hasattr(ssnmf, 'SSNMFDConfig'), "ssnmf.SSNMFDConfig not found"
    print("✓ ssnmf.SSNMFDConfig accessible")

    assert hasattr(ssnmf, 'SSNMF'), "ssnmf.SSNMF not found"
    print("✓ ssnmf.SSNMF accessible")

    assert hasattr(ssnmf, 'SSNMFD'), "ssnmf.SSNMFD not found"
    print("✓ ssnmf.SSNMFD accessible")
except AssertionError as e:
    print(f"❌ Accessibility check failed: {e}")
    sys.exit(1)

# Test 3: Create configs using old API
print("\n[TEST 3] Creating configs with old API")
try:
    base_config = ssnmf.BaseNMFConfig(
        n_components=5,
        max_iter=10,
        tol=1e-4,
        random_state=42,
        verbose=False,
    )
    print(f"✓ Created BaseNMFConfig: {base_config}")

    config = ssnmf.SSNMFDConfig(
        ssnmf_config=base_config,
        lambda_reg=0.1,
        gamma_reg=0.1,
        n_neighbors=3,
    )
    print(f"✓ Created SSNMFDConfig: {config}")
except Exception as e:
    print(f"❌ Config creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Instantiate and fit model with old API (with deprecation warning)
print("\n[TEST 4] Instantiating and fitting model with old API")
print("(Expect deprecation warnings)")
try:
    # Catch deprecation warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        model = ssnmf.SSNMFD(config)

        # Check that deprecation warning was issued
        assert len(w) > 0, "Expected deprecation warning"
        assert issubclass(w[0].category, DeprecationWarning), "Expected DeprecationWarning"
        print(f"✓ Deprecation warning issued: {w[0].message}")

    # Generate small test data
    np.random.seed(42)
    X = np.abs(np.random.randn(10, 15))

    # Fit model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings for actual fitting
        model.fit(X)

    print("✓ Model fitted successfully")
    print(f"  W shape: {model.W.shape}")
    print(f"  H shape: {model.H.shape}")
except Exception as e:
    print(f"❌ Model fitting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify new API also works
print("\n[TEST 5] New API also accessible")
try:
    from clustering_methods import SSNMFDFrobenius, create_ssnmfd_config
    print("✓ New imports successful")

    new_config = create_ssnmfd_config(n_components=5, max_iter=10, verbose=False)
    new_model = SSNMFDFrobenius(new_config)
    print("✓ New model instantiated")

    new_model.fit(X)
    print("✓ New model fitted successfully")
    print(f"  W shape: {new_model.W.shape}")
    print(f"  H shape: {new_model.H.shape}")
except Exception as e:
    print(f"❌ New API test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL BACKWARD COMPATIBILITY TESTS PASSED")
print("=" * 60)
print("\nConclusion:")
print("- Old code (like sample_ssnmf_d.py) will continue to work")
print("- Deprecation warnings will inform users to migrate to new API")
print("- New API provides improved functionality and proper label constraints")
