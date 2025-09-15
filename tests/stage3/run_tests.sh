#!/bin/bash
# Test runner for Stage 3 malformed reprocessing tests

echo "🧪 Running Stage 3 Malformed Reprocessing Tests"
echo "================================================="

cd "$(dirname "$0")"/../..

python3 tests/stage3/test_malformed_reprocessing.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ All tests passed! The malformed reprocessing logic is working correctly."
    echo "💡 It's safe to proceed with malformed entry reprocessing."
else
    echo ""
    echo "❌ Tests failed! Do not proceed with malformed entry reprocessing."
    echo "🔧 Review and fix the malformed reprocessing logic first."
fi

exit $exit_code