from mars.induction.self_induced_language import NumericTrace, SelfInducedLanguage


def test_self_induced_language_finds_arithmetic_fragment():
    xs = tuple(float(i) for i in range(1, 12))
    ys = tuple(3.0 * x * x + 2.0 for x in xs)

    result = SelfInducedLanguage(max_depth=2).induce(
        NumericTrace(name="quadratic", x=xs, y=ys)
    )

    assert result.programs
    assert result.programs[0].compression_gain > 1.0
    assert "x" in result.programs[0].expr


def test_self_induced_language_has_no_domain_function_seed():
    result = SelfInducedLanguage(max_depth=1).induce(
        NumericTrace(name="linear", x=(1.0, 2.0, 3.0, 4.0), y=(2.0, 4.0, 6.0, 8.0))
    )

    expressions = " ".join(p.expr for p in result.programs)
    assert "sin" not in expressions
    assert "acos" not in expressions
    assert "exp" not in expressions
