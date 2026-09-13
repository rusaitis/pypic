#!/usr/bin/env bash
# The checks CI gates on, over the paths CI actually covers.
#
# CI, CONTRIBUTING.md, the pull-request template and CLAUDE.md all point
# here rather than repeating the path list, which had already drifted:
# three of them claimed `src tests` while CI also covered scripts/,
# benchmarks/ and the committed examples, so touching those directories
# gave a green local run and a red pull request.
#
# ruff now takes the repo root instead of that list — see the note above
# lint() for why.
#
# Usage: scripts/check.sh [lint|format|types|test|cov|docs|schema]
#        scripts/check.sh            # everything, in CI order
#
# `cov` is a CI job of its own and deliberately outside the default
# chain: it runs the same suite `test` already ran, only instrumented,
# and costs about 60% more wall clock for it.
set -euo pipefail

cd "$(dirname "$0")/.."

# mypy is given explicit paths; it has no .gitignore awareness, and tests/
# is deliberately out (the suite imports examples/ as a package, which makes
# the same file resolvable under two module names, and strict mode over
# fixtures buys little). ruff still covers tests/.
MYPY_PATHS=(src scripts benchmarks examples)

# ruff gets the repo root, not a path list. Measured 2026-09-12 with ruff
# 0.15.5: a multi-root invocation (`ruff check src tests scripts benchmarks
# examples`) walks a different file set on each run — 264 files, or 69 with
# src/ dropped entirely — and caught a planted F401 in src/ once in ten
# runs. `--no-cache` does not help; it pins the walk to the truncated set.
# A single root is deterministic and a strict superset of the five paths.
lint()   { uv run ruff check .; }
format() { uv run ruff format --check .; }
types()  { uv run mypy "${MYPY_PATHS[@]}"; }
test()   { uv run pytest; }
cov()    { uv run pytest --cov --cov-report=term-missing; }
docs()   { uv run mkdocs build --strict; }

# Scoped to the generated artifact, not the whole schema/ directory: any
# unrelated edit to a schema module would otherwise read as drift.
schema() {
  local artifact=src/pypic/schema/simulation.schema.v2.0.json
  uv run pypic schema export -o "$artifact"
  git diff --exit-code -- "$artifact"
}

if [ $# -gt 0 ]; then
  "$1"
else
  lint && format && schema && types && test && docs
fi
