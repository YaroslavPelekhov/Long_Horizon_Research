import math

from mars.induction.operator_genome import (
    MeasurementTrace,
    induce_univariate_operators,
    infer_failure_geometry,
)


def test_phase_chart_is_born_from_oscillatory_trace():
    xs = tuple(i / 10 for i in range(1, 30))
    ys = tuple(math.sin(2 * x) for x in xs)

    trace = MeasurementTrace(name="toy_phase", xs=xs, ys=ys)
    geometries = infer_failure_geometry(trace)
    operators = induce_univariate_operators(trace)

    assert any(g.kind == "oscillatory_or_phase_residual" for g in geometries)
    assert operators
    assert operators[0].family == "phase_chart"
    assert operators[0].posterior_score > 1.0


def test_saturation_trace_births_inverse_coordinate_chart():
    xs = tuple(i / 10 for i in range(1, 40))
    ys = tuple(math.atan(x) for x in xs)

    trace = MeasurementTrace(name="toy_inverse", xs=xs, ys=ys)
    operators = induce_univariate_operators(trace)

    assert any(op.family == "inverse_coordinate_chart" for op in operators)
