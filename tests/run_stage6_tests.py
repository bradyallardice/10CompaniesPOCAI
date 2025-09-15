#!/usr/bin/env python3
"""
Stage 6 Test Runner - Runs all Stage 6 tests in sequence
"""

import sys
import subprocess
import os
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file and return results"""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v'
        ], cwd=Path(__file__).parent.parent, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False

def main():
    """Run all Stage 6 tests"""
    print("🧪 Stage 6 Test Suite Runner")
    print("Running comprehensive tests for Stage 6: Link Exposure to Jobs")
    
    # Test files in logical order
    test_files = [
        'tests/test_stage6_units.py',      # Unit tests first
        'tests/test_stage6_joins.py',      # Join logic tests
        'tests/test_stage6_failfast.py',   # Fail-fast validation tests
        'tests/test_stage6_cli.py',        # CLI integration tests
    ]
    
    results = {}
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file] = False
            total_failed += 1
            continue
            
        success = run_test_file(test_file)
        results[test_file] = success
        
        if success:
            print(f"✅ {test_file} - PASSED")
            total_passed += 1
        else:
            print(f"❌ {test_file} - FAILED")
            total_failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("STAGE 6 TEST SUMMARY")
    print('='*60)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file:<45} {status}")
    
    print(f"\n📊 Results: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("🎉 All Stage 6 tests passed!")
        return 0
    else:
        print(f"💥 {total_failed} test files failed")
        return 1

if __name__ == "__main__":
    exit(main())