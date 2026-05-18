r"""Speedup benchmark: batched vs per-seed adaptive field-line tracing.

Not part of CI. Run manually after changes to the Dormand-Prince kernel
or the tracer dispatch to verify the batched form still pays.

Usage
-----
    uv run python benchmarks/bench_batched_tracer.py

Method
------
A 64³ vortex field $\mathbf{B} = (-y, x, 0.2)$ on $[-1, 1]^3$ — smooth,
no nulls, seeds spiral with similar trajectory lengths so the batched
path isn't penalized by early-termination compute waste. Seeds are drawn
from a fixed-seed RNG, so successive runs on the same machine are
reproducible. Both paths use the same adaptive tolerances and the same
``VectorFieldInterpolator`` instance; the only difference is the
dispatch (N scalar calls vs one batched call).

This bench leaves ``loop_tol`` unset, so the always-on auto closed-loop
detector is active — the helix's per-orbit axial drift (~1.23 in
arclen) far exceeds the auto threshold (0.5 × grid spacing ≈ 0.016),
so the detector runs every step but never fires. The numbers below
therefore include the per-step cost of the vectorized arclen update
and proximity scan; subtract a few percent at N=1000 if you want the
pure-kernel timing (pass ``loop_tol=None``).

Expected numbers
----------------
On a modern workstation, single-threaded NumPy, you should see roughly:

         N    scalar (s)   batched (s)   speedup
    ----------------------------------------------
         1        ~0.007        ~0.007    ~1.0x
        10        ~0.08         ~0.018    ~4-5x
       100        ~0.85         ~0.045    ~15-20x
      1000        ~8.5          ~0.25     ~30-35x

The curve flattens past N ~ 10³ because the irreducible per-point
interpolator cost takes over — Python/dispatch overhead is fully
amortized by then. Wall-clock numbers will vary with CPU, BLAS, NumPy
version, and noise from co-tenants; speedup ratios should remain
stable to within ±20% on the same machine.

A regression to ~10× at N=1000 (with otherwise unchanged correctness
tests) is a strong signal that something cheap was lost — FSAL re-use,
tensordot batching, the interpolator's vector dispatch — and worth
bisecting.
"""

from __future__ import annotations

import time

import numpy as np

from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.traces import trace_field_line_adaptive, trace_field_lines_adaptive
from pypic.units import Normalization


def build_vortex_field(n: int = 64) -> FieldDataset:
    r"""$\mathbf{B} = (-y, x, 0.2)$ on $[-1, 1]^3$ as an n × n × n grid."""
    coords = np.linspace(-1.0, 1.0, n)
    x, y, _ = np.meshgrid(coords, coords, coords, indexing="ij")
    return FieldDataset.from_arrays(
        {"B_1": -y, "B_2": x, "B_3": 0.2 * np.ones_like(x)},
        GridInfo(
            dimensions=(n, n, n),
            spacing=(2.0 / (n - 1), 2.0 / (n - 1), 2.0 / (n - 1)),
            origin=(-1.0, -1.0, -1.0),
        ),
        Normalization.identity(),
    )


def random_seeds(n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """N seeds in $[-0.5, 0.5]^3$ (well inside the $[-1, 1]^3$ domain)."""
    return rng.uniform(-0.5, 0.5, size=(n_seeds, 3))


_TRACER_KW: dict[str, float | int | str] = {
    "atol": 1e-6,
    "rtol": 1e-6,
    "step_size_init": 0.02,
    "min_step": 1e-6,
    "max_step": 0.1,
    "max_steps": 500,
    "direction": "forward",
}


def time_scalar(data: FieldDataset, seeds: np.ndarray) -> float:
    """Per-seed scalar adaptive trace over N seeds. Returns wall-clock seconds."""
    t0 = time.perf_counter()
    for i in range(seeds.shape[0]):
        trace_field_line_adaptive(data, tuple(seeds[i].tolist()), **_TRACER_KW)  # type: ignore[arg-type]
    return time.perf_counter() - t0


def time_batched(data: FieldDataset, seeds: np.ndarray) -> float:
    """One batched adaptive trace over N seeds. Returns wall-clock seconds."""
    t0 = time.perf_counter()
    trace_field_lines_adaptive(data, seeds, **_TRACER_KW)  # type: ignore[arg-type]
    return time.perf_counter() - t0


def main() -> None:
    """Run the warmup pass then time scalar vs batched at N = 1, 10, 100, 1000."""
    rng = np.random.default_rng(42)
    data = build_vortex_field(n=64)

    # Warmup: prime interpolator caches, NumPy import overhead, page faults.
    # Without this the first measured N=1 row is artificially slow.
    warmup_seeds = random_seeds(4, rng)
    time_scalar(data, warmup_seeds)
    time_batched(data, warmup_seeds)

    print(f"{'N':>6}  {'scalar (s)':>12}  {'batched (s)':>12}  {'speedup':>8}")
    print("-" * 46)

    for n_seeds in [1, 10, 100, 1000]:
        seeds = random_seeds(n_seeds, rng)
        t_scalar = time_scalar(data, seeds)
        t_batch = time_batched(data, seeds)
        speedup = t_scalar / t_batch if t_batch > 0 else float("inf")
        print(f"{n_seeds:>6}  {t_scalar:>12.4f}  {t_batch:>12.4f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
