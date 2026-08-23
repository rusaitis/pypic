# Field Line Tracing

Curve containers and analysis for field lines and particle trajectories. `trace_field_line` integrates with fixed-step classical RK4;
`trace_field_line_adaptive` uses Dormand-Prince 5(4) with error-norm step
control. Both feed the same geometric analysis (curvature, arc length,
mirror points).

::: pypic.traces
