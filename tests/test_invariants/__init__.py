"""Machine-checked physics and API identities, as Hypothesis property tests.

The flat modules under ``tests/`` are worked examples and regressions:
a specific input, a specific expected number, often hand-calculated or
taken from the NRL Formulary. The modules in *this* package are the
complementary half — statements that must hold for *every* input in a
generated domain, checked with ``hypothesis`` over strategies from
``tests/strategies.py``.

Two conventions apply here and nowhere else:

* Every module opens with a ``# Source:`` line naming the code or
  documentation the identity comes from, and a ``# Claim:`` line stating
  the property in one sentence. A property test whose claim is not
  traceable to a formula is a test nobody can review.
* ``deadline`` is set once, in ``tests/conftest.py``, via the ``pypic``
  Hypothesis profile. Do not re-declare it per test.

Add a module here when the assertion is "for all X, P(X)". Add one to
the flat suite when it is "for this X, the answer is 3.7".
"""
