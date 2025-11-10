#!/usr/bin/env python3
"""
Comprehensive test runner for Stage 6
Runs all Stage 6 tests including both new and existing tests together
"""

import subprocess
import sys
from pathlib import Path

def run_all_stage6_tests():
    """Run all Stage 6 tests in a single pytest command"""

    test_dir = Path(__file__).parent

    print("🧪 Running Comprehensive Stage 6 Test Suite")
    print(f"Test directory: {test_dir}")
    print("="*60)

    # Run all Stage 6 tests together
    stage6_test_files = [
        # New comprehensive tests
        "test_stage6_real_files.py",
        "test_stage6_cache.py",
        "test_stage6_scale.py",
        "test_stage6_e2e.py",

        # Existing tests
        "test_stage6_units.py",
        "test_stage6_joins.py",
        "test_stage6_cli.py",
        "test_stage6_failfast.py"
    ]

    # Filter to only existing files
    existing_files = []
    for test_file in stage6_test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            existing_files.append(str(test_path))
        else:
            print(f"⚠️  Test file not found (skipping): {test_file}")

    if not existing_files:
        print("❌ No test files found!")
        return 1

    print(f"Running {len(existing_files)} Stage 6 test files...")
    print()

    # Run all tests together with pytest
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest"
        ] + existing_files + [
            "-v",
            "--tb=short",
            "--color=yes",
            "--durations=10",  # Show slowest 10 tests
            "--strict-markers",
            "--strict-config"
        ], cwd=test_dir)

        return result.returncode

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_all_stage6_tests()
    if exit_code == 0:
        print("\n🎉 All Stage 6 tests completed successfully!")
    else:
        print(f"\n⚠️  Tests completed with exit code: {exit_code}")
    sys.exit(exit_code)