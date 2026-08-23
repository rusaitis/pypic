#!/usr/bin/env bash
# The checks CI gates on, over the paths CI actually covers.
#
# CI, CONTRIBUTING.md, the pull-request template and CLAUDE.md all point
# here rather than repeating the path list, which had already drifted:
# three of them claimed `src tests` while CI also covered scripts/,
# benchmarks/ and the committed examples, so touching those directories
# gave a green local run and a red pull request.
#
# Usage: scripts/check.sh [lint|format|types|test|docs|schema]
#        scripts/check.sh            # everything, in CI order
set -euo pipefail

cd "$(dirname "$0")/.."

# Every Python path CI lints and type-checks. examples/ is globbed: only
# committed scripts live there, since .gitignore keeps local simulation
# data out (see the examples/ block there).
PATHS=(src tests scripts benchmarks examples)

# mypy skips tests/: the suite imports examples/ as a package, which makes
# the same file resolvable under two module names, and strict mode over
# fixtures buys little. ruff still covers it.
MYPY_PATHS=("${PATHS[@]/tests}")

lint()   { uv run ruff check "${PATHS[@]}"; }
format() { uv run ruff format --check "${PATHS[@]}"; }
types()  { uv run mypy ${MYPY_PATHS[@]}; }
test()   { uv run pytest; }
docs()   { uv run mkdocs build --strict; }

# Scoped to the generated artifact, not the whole schema/ directory: any
# unrelated edit to a schema module would otherwise read as drift.
schema() {
  local artifact=src/pypic/schema/simulation.schema.v1.0.json
  uv run pypic schema export -o "$artifact"
  git diff --exit-code -- "$artifact"
}

if [ $# -gt 0 ]; then
  "$1"
else
  lint && format && schema && types && test && docs
fi
