# pypic Invariant Auto-Researcher

Autonomous loop that hunts for bugs in pypic by writing **Hypothesis
properties** against invariants the codebase already promises — in
`docs/schema.md`, `CLAUDE.md`, `docs/equations.md`, and public docstrings.

The loop replaces a scalar fitness function (like the QP-detection
composite score) with a **discrete, asymmetric signal**: a Hypothesis
counterexample is always informative — either the code has a bug, or a
stated invariant was wrong. Every iteration moves the suite forward.
The loop cannot no-op-commit.

## Goal

Each iteration: pick one untested invariant, write a Hypothesis property
against it, run it. Commit one of two outcomes:

- **Counterexample found** → minimize, diagnose, fix the bug. Commit the
  regression test together with the fix. `fix:` prefix.
- **No counterexample under the examples budget** → commit the property
  alone. It strengthens the suite (measurable later as mutation-kill
  uplift). `test:` prefix.

## Philosophy

Prefer invariants with **a citable source AND a named bug class**. If
you cannot point at a line in `schema.md`, `CLAUDE.md`, `equations.md`,
or a public docstring that the property encodes, the property is not
worth testing — it is speculation dressed up as a test. Every property
file has a **two-line header**: (a) source citation, (b) `Catches:
<concrete bug class>`. Vague catches like "any sign error" or "any
refactor that..." don't count — name a specific shape of mistake
(a wrong CGL exponent, a missing factor of 1/2 between two formulas,
a degenerate-input overshoot, an axis-aligned coordinate leak, a
silent reader/registry name mismatch). If you can't name the bug class
concretely, the property isn't worth testing.

**Good invariants**

- Round-trips the code claims to preserve (`to_si(normalize(x)) == x`,
  Zarr/Parquet read→write→read bit-equality on metadata).
- Conservation laws the numerics promise (`div(curl F) ≈ 0` to machine
  precision on the stated stencil order).
- Algebraic identities between registered quantities
  (`compute("|B|")² == B1² + B2² + B3²`).
- API contracts stated in prose (`fields=["typo"]` raises `KeyError`,
  not silent skip).
- Physical limits (`c → ∞` in a relativistic formula agrees with the
  non-relativistic formula, scaling as `1/c²`).

**Bad invariants**

- "Looks right to me" — no written source. Skip.
- Tolerances tuned to make Hypothesis pass. If `rtol=1e-3` is needed
  for a first-order-accurate stencil on a smooth field, the stencil is
  broken, not the test.
- Band-/reader-/codepath-specific carve-outs. A property that holds for
  `iPIC3D_Parallel` but not the serial reader describes a bug in one of
  them — write both, commit both, let the loop find the divergence.

## Fitness function (keep/discard)

| Outcome | Action |
|---|---|
| Property falsifies; source claim is real; code has the bug | Commit fix + test together. `fix:` |
| Property passes; ruff + mypy + full suite green | Commit test alone. `test:` |
| Property falsifies; stated claim was not actually promised | Revert. Scoreboard entry: `ill-posed`. Do not weaken the property to make it pass. |
| Property falsifies only on degenerate input (zero-size grid, all-NaN array) | Minimally shrink the strategy, cite the degenerate case in the strategy docstring, re-run. Flag the loop if this happens twice consecutively. |
| Full suite regresses after the "fix" | Revert. The fix was wrong. |

**Hard rules** (non-negotiable, analogous to "no band-specific
thresholds" in the QP loop):

- No loosening of existing tolerances to accommodate a new property.
- No `@pytest.mark.xfail` without a source-linked comment naming which
  claim is known-broken and why the fix is out of scope.
- Every new property file opens with the **two-line header** required
  by § "Philosophy": (a) source citation
  (`# schema.md § 4`, `# CLAUDE.md "fail loud on unmatched names"`,
  `# equations.md footnote [^9]`, `# units.py:309 docstring`) and
  (b) `# Catches: <concrete bug class>`.
- Fixes must not change the on-disk schema (schema.md § 4) or break any
  existing Zarr/Parquet/Arrow round-trip test.
- Strategy shrinks are a yellow flag: ≥2 consecutive iterations that
  needed one → stop the loop and surface to the user.

## Invariant backlog (ordered by ROI)

The loop works top-down. **When this list is exhausted, halt and
surface to the user.** Restart only when a source-of-truth artifact
changes: `docs/schema.md`, `docs/equations.md`, `docs/conventions.md`,
`CLAUDE.md`, or a new step ships in `TASKS.md` that introduces a
derived function with a non-trivial docstring claim. The loop is a
*responder* to documented contracts, not a generator of them.

1. **Normalization round-trip on all four systems × all base quantities.**
   `tests/test_units.py:65` covers `pic_electron` + `mhd_standard` only.
   Extend to `pic_standard` and `identity`. Source: `docs/schema.md § 2`.
2. **`compute(name)` dispatch == direct function call** for every entry in
   the derived-quantity registry. `tests/test_compute.py:406` covers one
   case. Source: CLAUDE.md "registered quantities" + `src/pypic/compute.py`.
3. **`div(curl F) ≈ 0` on random smooth fields** below Nyquist. Source:
   `docs/equations.md § 6`, second-order central-difference docstring.
4. **Frame-transform associativity** (inverse is already tested at
   `tests/test_transforms.py:67`). Source: CLAUDE.md transform-chaining rule.
5. **`ParticleData.macro_charge == species_charge * weight`** bit-exact.
   Source: `docs/schema.md § 3` per-particle columns + TASKS.md Step 25b.
6. **Zarr/Parquet metadata fidelity under `dtype=float32` downcast.** Field
   arrays may lose precision; `Normalization`, `StaggerInfo`, species
   scalars must round-trip bit-exact. Source: TASKS.md Step 24 / Step 25.
7. **Relativistic c → ∞ limit.** For every derived function with
   `c: float | None`, the `c=1e12` branch must agree with `c=None` within
   tolerance scaling as `1/c²`. Source: CLAUDE.md "relativistic via
   `c=None` kwarg" + `docs/equations.md § 8`.
8. **Selection APIs raise `KeyError` on unmatched names, never log-and-skip.**
   Source: CLAUDE.md "fail loud on unmatched names".
9. **Vector-group shorthand idempotence.** `"B"` → `["B1","B2","B3"]`;
   expanding twice equals expanding once; unknown names raise. Source:
   `docs/schema.md § 3` "Vector group shorthand in `read()`".

## Iteration contract

1. Read `scoreboard-invariants.tsv`. Pick the next unchecked backlog
   item. **If the backlog is exhausted, halt and surface to the user**
   (see § "Invariant backlog" — the loop does not manufacture fresh
   invariants).
2. Write the property at `tests/test_invariants/test_<cluster>.py`.
   Start the file with the **two-line header** required by §
   "Philosophy": (a) source citation, (b) `Catches: <concrete bug
   class>`. Reuse the strategies in `tests/strategies.py` and the
   factories in `tests/_helpers.py` — do not duplicate them.
   - **Sub-test budget**: ≤2 `@given` functions per file. Bundle
     multiple facets of one identity into a single parametrized test
     (iter 23's CGL-exponent test is the reference shape: one
     function, three algebraic checks). If a third `@given` looks
     necessary, drop the weakest or split into a separate iteration.
3. Run just the new file first:
   ```
   uv run pytest tests/test_invariants/test_<cluster>.py --hypothesis-seed=random -x
   ```
4. Handle the outcome per the fitness-function table above. If
   committing, run the full suite plus lint:
   ```
   uv run pytest -v
   uv run ruff check src tests
   uv run mypy src
   ```
   All three must be green.
5. Append one row to `scoreboard-invariants.tsv` (including the
   `sub_tests` column count for this iteration) AND identify the
   weakest existing invariant test. Either delete it in the same
   commit, or write one sentence in the scoreboard `notes` column
   explaining why every existing test still pulls weight. The total
   invariant test count must not grow monotonically — adding requires
   either replacing or justifying. Commit message format:
   - Bug: `fix: <short why, not what>` + `Assisted-by: Claude Opus 4.6 (1M context)`.
   - Property: `test: hypothesis property for <invariant>` + same
     suffix. If the iteration also pruned, name the dropped test in
     the body.
6. Halt after any of:
   - 10 iterations
   - 2 consecutive `ill-posed` outcomes
   - 2 consecutive `strategy-shrink` events
   - **10 consecutive `test-added` iterations with no `bug-fixed` or
     `bug-surfaced` outcome.** The loop's purpose is bug hunting; if
     10 in a row produce only properties, the marginal value is below
     the maintenance cost. Surface to the user with a recommendation
     to switch to mutation testing (`mutmut` / `cosmic-ray`) and let
     surviving mutants drive the next round of properties.

## What you CAN modify

| File | When |
|---|---|
| `tests/test_invariants/*.py` | Every iteration — new properties land here. |
| `tests/strategies.py` | When a new invariant needs a strategy not yet in the catalog. Add, don't mutate existing strategies. |
| `src/pypic/**/*.py` | Only when a counterexample reveals a real bug. Minimal fix; preserve the public API. |
| `scoreboard-invariants.tsv` | Append one row per iteration (with `sub_tests` count). |
| `tests/test_invariants/*.py` (deletes) | Iteration step 5 prunes the weakest existing test alongside any add. |

## What you CANNOT modify

- `docs/schema.md`, `docs/equations.md`, `docs/conventions.md`,
  `CLAUDE.md` — these are the **source of truth** the loop reads. If a
  claim there is wrong, surface it to the user; do not rewrite it.
- Existing tests (`tests/test_*.py` outside `test_invariants/`) — the
  loop adds, it does not mutate.
- On-disk schema (HDF5/Zarr/Parquet layouts). A "fix" that requires
  changing the file format is out of scope.
- `pyproject.toml` dependency versions. No new deps beyond what's
  already declared.
