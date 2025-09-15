#!/bin/bash
# Test runner for Stage 5 pipeline tests

echo "🧪 Running Stage 5 AI Exposure Calculation Tests"
echo "================================================="

cd "$(dirname "$0")/.."

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Please install with: pip install pytest"
    exit 1
fi

# Run the tests with verbose output
pytest tests/test_stage5.py -v --tb=short

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ All Stage 5 tests passed! The exposure calculation pipeline is working correctly."
    echo "💡 Safe to proceed with Stage 5 processing on real data."
else
    echo ""
    echo "❌ Some Stage 5 tests failed! Review the pipeline logic before processing real data."
    echo "🔧 Check the test output above for specific failure details."
fi

exit $exit_code