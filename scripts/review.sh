#!/bin/bash
# Pre-commit review script for TDD cycle
# Run before each commit to catch suspicious changes

set -e

echo "=== PRE-COMMIT REVIEW ==="
echo ""

# Dependencies for tests
DEPS="--with pytest --with sqlite-vec --with openai"

# 1. Test count check
echo "1. Test count:"
CURRENT_TESTS=$(uv run $DEPS pytest --collect-only -q 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1)
echo "   Current: $CURRENT_TESTS tests"

# 2. All tests pass
echo ""
echo "2. Test results:"
if uv run $DEPS pytest -q 2>/dev/null; then
    echo "   ✓ All tests pass"
else
    echo "   ✗ TESTS FAILING - DO NOT COMMIT"
    exit 1
fi

# 3. Suspicious patterns in staged changes
echo ""
echo "3. Suspicious patterns in diff:"
SUSPICIOUS=$(git diff --cached -- '*.py' | grep -E '^\+.*(pytest\.mark\.skip|pytest\.skip|@skip|assert True.*#|assert False.*#|pass\s*#.*test)' || true)
if [ -n "$SUSPICIOUS" ]; then
    echo "   ⚠ WARNING: Found suspicious patterns:"
    echo "$SUSPICIOUS"
else
    echo "   ✓ No suspicious patterns"
fi

# 4. Deleted test functions
echo ""
echo "4. Deleted test functions:"
DELETED_TESTS=$(git diff --cached -- '*.py' | grep -E '^\-\s*def test_' || true)
if [ -n "$DELETED_TESTS" ]; then
    echo "   ⚠ WARNING: Tests deleted:"
    echo "$DELETED_TESTS"
else
    echo "   ✓ No tests deleted"
fi

# 5. Summary of changes
echo ""
echo "5. Changes summary:"
git diff --cached --stat

echo ""
echo "=== REVIEW COMPLETE ==="
