"""Pure-numerics kernels for pypic.

Generic ODE integrators, step controllers, and related numerical
methods, decoupled from any specific physics task. Today the only
consumer is the adaptive field-line tracer
(``pypic.traces.trace_field_line_adaptive``); future consumers
(particle pushers, splitting helpers, higher-order quadrature) land
here so they don't need to be re-extracted from their first caller.

Public surface intentionally minimal — not re-exported from the
top-level ``pypic`` namespace until a second consumer emerges.
"""

from pypic.numerics._rk import (
    DPStepResult,
    DPStepResultBatched,
    dormand_prince_step,
    dormand_prince_step_batched,
    embedded_error_norm,
    embedded_error_norm_batched,
)
from pypic.numerics._step_control import (
    i_step_controller,
    i_step_controller_batched,
)

__all__ = [
    "DPStepResult",
    "DPStepResultBatched",
    "dormand_prince_step",
    "dormand_prince_step_batched",
    "embedded_error_norm",
    "embedded_error_norm_batched",
    "i_step_controller",
    "i_step_controller_batched",
]
