# pypic Test-Curation Auto-Researcher

Companion loop to `autoresearcher-pypic.md`. Where loop 1 *generates*
new Hypothesis properties from documented invariants, this loop
*audits* the existing invariant tests — pruning bloat, hardening weak
assertions, and giving every kept test a one-line written
justification.

The signal here is **LLM judgment under tight constraints**, not
falsifiable counterexamples. To stop the same bloat that halted loop
1, every action must produce either a code diff or a written reason
the test still pulls weight. Rubber-stamp `kept` outcomes are
forbidden.

## Goal

Each iteration: review the next ~50 unreviewed tests in
`tests/test_invariants/`. For each, commit one of:

- **Drop** the test with written rationale (≥1 sentence in the
  commit body) showing coverage isn't lost. `test:` prefix.
- **Harden** the test: tighten an assertion, add a parametrize case,
  expand a Hypothesis strategy domain, fix a stale comment. `test:`
  prefix.
- **Split / merge** when one test has two intents or two tests test
  one thing. `test:` prefix.
- **Keep** with a non-vague one-line justification recorded in
  `scoreboard-tests.tsv`. Bulk `kept`-only iterations get a single
  `chore:` commit.
- **Surface to loop 1** when review reveals a doc-coverage gap (emits
  `plans/loop1-candidate-*.md`, no code change).
- **Surface a bug** when review reveals a defect in `src/pypic/**`
  (emits a note to the user, no code change in this loop).

## Philosophy

Loop 1's bloat (audited at iters 21 and 32) accumulated because the
contract weighted *adding* over *removing*: a passing property test
shipped, a failing one became a fix; nothing made the loop reckon
with whether an existing test was still pulling weight. This loop
inverts that asymmetry. Reviewing a test produces exactly one of
three useful artifacts: a deletion, a strengthening, or a written
reason for existence. Anything else is a rubber stamp.

**Vague justifications are rejected.** The bar mirrors loop 1's
"don't claim a `Catches:` you can't name concretely":

- ✗ "Tests basic behavior" / "covers happy path" / "looks fine"
- ✗ "Sanity check" / "regression guard"
- ✓ "Guards against the catastrophic-cancellation rewrite of
  kinetic_energy_density (iter 7) — would silently revert under a
  copy-paste from the textbook formula."
- ✓ "Pins the per-species charge sign convention from
  iPIC3D (`q_e < 0`, `q_i > 0`); flipping it produces a
  same-magnitude J that passes magnitude tests."

If you can't name a concrete bug class, mistake shape, or convention
the test pins, the test is `dropped` or `hardened` — not `kept`.

**Drops are first-class outputs.** A loop iteration that drops zero
tests is a yellow flag (one in isolation is fine; three in a row
halts). The invariant test count must not grow monotonically.

**Source bugs surface, they don't get fixed.** If a review reveals
`src/pypic/**` is wrong, emit a `surfaced-bug` note and let the user
fix it. This loop does not edit source — that boundary keeps the
audit trail clean (a `test:` commit never bundles unrelated source
changes) and prevents loop 2 from drifting into territory that needs
its own contract.

## Fitness function (keep / discard)

| Outcome | Action |
|---|---|
| Test names a concrete bug class it pins; assertion strength matches the claim | `kept` with one-line justification recorded. No commit per test (bulk `chore:` commit per iteration if all-kept). |
| Test guards a real claim but assertion is weaker than the docstring promises | `hardened`: tighten assertion / add parametrize case / expand strategy. `test:` |
| Test's killed-mutant set (informally judged) is a strict subset of another test's | `dropped` with overlap proof in commit body. `test:` |
| One `@given` function bundles two distinct algebraic claims | `split` into two clearer functions. `test:` |
| N tests are copy-paste variants on a shared identity | `merged` into one parametrized form. `test:` |
| Test's claim has no source in `docs/equations.md` / `docs/schema.md` / `docs/conventions.md` / `CLAUDE.md` and no concrete bug class can be named | `dropped` with rationale "no source claim, no nameable bug class — speculation dressed as a test." `test:` |
| Review surfaces a doc-coverage gap (a real invariant the docs don't promise) | `surfaced-to-loop-1`: emit `plans/loop1-candidate-*.md`. No code change. |
| Review surfaces a `src/pypic/**` bug | `surfaced-bug`: surface to user. No code change in this loop. |
| Full suite regresses or lint/mypy fails after a `dropped`/`hardened`/`split`/`merged` | Revert. The action was wrong. |

**Hard rules** (non-negotiable, mirror loop 1's spirit):

- **Every kept test gets a non-vague one-line justification.** See
  the Philosophy examples for the bar. Vague is rejected.
- **Drops require written rationale (≥1 sentence in commit body)**
  explaining why coverage isn't lost. If unsure, harden instead of
  drop.
- **No tolerance loosening** to make a test "pass review" or to keep
  a flaky test green. The math either supports tightening or the
  test is `dropped`.
- **Never delete the only test for a public function.** Verify via
  `grep -rn 'def <fn_name>' src/pypic/` and
  `grep -rn '<fn_name>' tests/`. If the test under review is the
  sole guard, the outcome must be `hardened` or `kept` — never
  `dropped`.
- **Cannot mutate loop 1's two-line `Source:` / `Catches:` header**
  in `tests/test_invariants/*.py`. Hardening test bodies is allowed;
  rewriting the header is not.
- **A commit must touch at least one test file or
  `scoreboard-tests.tsv`.** Pure-config commits don't qualify.
- **Splits/merges must preserve assertion count.** A split that
  drops an assertion in transit is two outcomes (`split` +
  `dropped`) on two ledger rows, not one.
- **Strategy shrinks** (loop 1's signal) are out of scope here —
  this loop doesn't run Hypothesis differently; if a hardening
  introduces a strategy-shrink, revert and pick a different
  hardening.

## Iteration contract

1. Read `scoreboard-tests.tsv`. Pick the next ~50 rows where
   `outcome == "not-yet-reviewed"` (file-order). If none remain,
   **halt and surface to the user** — Phase 1 is complete.
2. For each test in the batch:
   - Read the test body.
   - Read the source function it covers (`src/pypic/<module>`).
   - Read any invariant claim cited in the file's two-line header
     (`Source:` line — `docs/equations.md § X`, `CLAUDE.md "..."`, etc.).
   - Judge outcome per the fitness-function table.
3. Apply edits in one diff per file. Group all hardenings/drops in
   `test_X.py` into one batch. Splits/merges may touch multiple
   files; record each touched test in its own ledger row.
4. Run, in this order:
   ```
   uv run pytest -v
   uv run ruff check src tests
   uv run mypy src
   ```
   All three must be green. If `pytest` regresses, revert the action
   that caused it and pick a different outcome for that test.
5. Update one row per reviewed test in `scoreboard-tests.tsv`
   (in-place edit by `test_id` key). Set `iter_reviewed`, `outcome`,
   `justification`, `commit_hash` (use `<pending>` until step 6
   commits, then amend with the real SHA — or commit the ledger
   update in a follow-up `chore:` commit if the action commit needs
   to land first).
6. Commit. Two shapes:
   - **Mixed iteration** (any `hardened`/`dropped`/`split`/`merged`):
     `test: curation pass N — M kept, K hardened, D dropped` with body
     listing dropped tests by name + 1-sentence justification each,
     plus hardened tests with one-line "what changed."
   - **All-kept iteration**: `chore: curation pass N — 50 kept` with
     body containing the 50 justifications grouped by file. Suspicious
     — see halt criteria.
   Suffix every commit: `Assisted-by: Claude Opus 4.6 (1M context)`.
7. **Halt** after any of:
   - 10 iterations.
   - All Phase 1 tests have `iter_reviewed > 0` (natural completion).
   - **3 consecutive iterations with 0 drops AND 0 hardens** (signal
     exhausted — every test rubber-stamped `kept`. Surface to user;
     possibly time for mutation testing).
   - **1 iteration where >50% of the batch is `dropped`** (deep bloat
     zone — surface for sanity check before continuing).
   - **>2 `surfaced-bug` outcomes in a single iteration** (real
     defects accumulating faster than curation can keep up — switch
     to bug fixing, this loop can wait).

## What you CAN modify

| File | When |
|---|---|
| `tests/test_invariants/**/*.py` (test bodies) | Every iteration — hardening, dropping, splitting, merging. |
| `scoreboard-tests.tsv` | Every iteration — one row per reviewed test. |
| `plans/loop1-candidate-*.md` | When a `surfaced-to-loop-1` outcome fires. |

## What you CANNOT modify

- `docs/schema.md`, `docs/equations.md`, `docs/conventions.md`,
  `CLAUDE.md`, `autoresearcher-pypic.md` — loop 1's source-of-truth
  and contract. If a review finds a doc claim is wrong, surface it
  via `surfaced-to-loop-1`; do not rewrite.
- `scoreboard-invariants.tsv` — loop 1's per-iteration log. Loop 2
  has its own ledger.
- `src/pypic/**/*.py` — loop 2 reviews tests, not source. Source
  bugs surface as `surfaced-bug` notes; the user fixes them.
- The two-line `Source:` / `Catches:` header in
  `tests/test_invariants/*.py` (loop 1's contract). Hardening test
  bodies is allowed; header rewrites are not.
- `tests/test_*.py` outside `test_invariants/` during Phase 1.
  Phase 2 needs explicit user approval to start.
- `pyproject.toml` dependency versions. No new deps beyond what's
  already declared.
