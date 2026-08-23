"""Pure-numerics kernels for pypic.

Generic ODE integrators, step controllers, and related numerical
methods, decoupled from any specific physics task. The adaptive
field-line tracer (``pypic.traces.trace_field_line_adaptive``) is the
current consumer. Not re-exported from the top-level ``pypic``
namespace.

Structure-preserving (symplectic, variational) integrators follow
[@HairerLubichWanner2006]; the current Dormand-Prince kernel is the
classical embedded pair from [@HairerWanner1993].
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
