#!/bin/bash
# Test runner for Stage 0 keyword extraction tests

echo "🧪 Running Stage 0 AI Keyword Extraction Tests"
echo "=============================================="

cd "$(dirname "$0")/.."

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Please install with: pip install pytest"
    exit 1
fi

# Run the tests with verbose output
pytest tests/test_stage0.py -v --tb=short

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ All Stage 0 tests passed! The keyword extraction logic is working correctly."
    echo "💡 Safe to proceed with Stage 0 batched keyword extraction on real data."
else
    echo ""
    echo "❌ Some Stage 0 tests failed! Review the keyword matching logic before processing real data."
    echo "🔧 Check the test output above for specific failure details."
fi

exit $exit_code