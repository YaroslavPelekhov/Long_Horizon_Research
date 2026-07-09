import unittest

from mars.skills import HypothesisSketch, ResidualKernel, compress_rewrite_rules


class ResidualKernelTests(unittest.TestCase):
    def test_induces_numeric_power_law_from_residuals(self):
        examples = []
        for m1, m2, distance in [
            (3.0, 10.0, 2.0),
            (10.0, 10.0, 2.0),
            (30.0, 10.0, 2.0),
            (10.0, 10.0, 1.0),
            (10.0, 10.0, 4.0),
            (100.0, 120.0, 8.0),
        ]:
            target = 6.674e-5 * m1 * m2 / (distance ** 1.5)
            examples.append(
                {
                    "mass1": m1,
                    "mass2": m2,
                    "distance": distance,
                    "target": target,
                }
            )

        program = ResidualKernel().induce_numeric_power_rule(examples)
        self.assertIsNotNone(program)
        assert program is not None
        powers = program.rule.residual.stats["powers"]
        self.assertAlmostEqual(powers["mass1"], 1.0)
        self.assertAlmostEqual(powers["mass2"], 1.0)
        self.assertAlmostEqual(powers["distance"], -1.5)

        ns = {"__builtins__": {"float": float, "abs": abs, "dict": dict, "str": str}}
        exec(
            program.code,
            ns,
        )
        law = ns["law"]
        pred = law({"mass1": 5.0, "mass2": 7.0, "distance": 4.0})
        expected = 6.674e-5 * 5.0 * 7.0 / (4.0 ** 1.5)
        self.assertLess(abs(pred - expected), 1e-9)

    def test_induces_context_pair_string_rules(self):
        examples = []
        for main, vice in [
            ("ABCDE", "EDCBA"),
            ("EDCBA", "ABCDE"),
            ("AABCE", "DDEAC"),
        ]:
            target = "".join(chr(ord("A") + ((ord(a) - ord("A") + ord(b) - ord("A")) % 26)) for a, b in zip(main, vice))
            examples.append(
                {
                    "current": "",
                    "target": target,
                    "context": {"main": main, "vice": vice},
                }
            )

        programs = ResidualKernel().induce_string_context_rules(examples)
        names = {p.name for p in programs}
        self.assertIn("rk_charwise_add_main_vice", names)
        program = [p for p in programs if p.name == "rk_charwise_add_main_vice"][0]
        ns = {}
        exec(
            program.code,
            {"__builtins__": {"str": str, "zip": zip, "ord": ord, "chr": chr, "range": range, "dict": dict}},
            ns,
        )
        rule = ns["rule"]
        self.assertEqual(rule("", {"main": "ABCDE", "vice": "EDCBA"}), "EEEEE")

    def test_induces_evidence_plan_slots(self):
        columns = {
            "Adjusted savings: education expenditure (% of GNI)": "government investment in education and human capital",
            "Labor force participation rate": "labor force proxy for human capital",
            "GNI per capita": "economic output and income per person",
            "Exports growth": "external output proxy",
        }
        plan = ResidualKernel().induce_evidence_plan(
            question="How does increased education expenditure influence human capital and economic output?",
            column_descriptions=columns,
            domain_context="Labor force is a proxy for human capital. GNI per capita represents economic output.",
        )
        self.assertIn("education", plan.cause.lower())
        self.assertTrue(any("labor" in item.lower() for item in plan.mediators))
        self.assertTrue(any("gni" in item.lower() for item in plan.outcomes))
        self.assertIn("mediation_chain", plan.operations)

    def test_compiles_table_evidence_sketch_to_program(self):
        import pandas as pd

        columns = {
            "Adjusted savings: education expenditure (% of GNI)": "government investment in education and human capital",
            "Labor force participation rate": "labor force proxy for human capital",
            "GNI per capita": "economic output and income per person",
            "Exports growth": "external output proxy",
        }
        kernel = ResidualKernel()
        sketch = kernel.induce_table_evidence_sketch(
            question="How does increased education expenditure influence human capital and economic output?",
            column_descriptions=columns,
            domain_context="Labor force is a proxy for human capital. GNI per capita represents economic output.",
        )
        self.assertIsInstance(sketch, HypothesisSketch)
        assert sketch is not None
        self.assertEqual(sketch.target_type, "table_evidence")
        expansion = kernel.expand_hypothesis_sketch(sketch)
        self.assertEqual(len(expansion.programs), 1)
        program = expansion.programs[0]
        ns = {"pd": pd, "__builtins__": {"abs": abs, "dict": dict, "float": float, "len": len, "range": range, "round": round, "set": set, "str": str, "sum": sum}}
        exec(program.code, ns)
        analyze = ns["analyze"]
        df = pd.DataFrame(
            {
                "Adjusted savings: education expenditure (% of GNI)": [1, 2, 3, 4, 5, 6],
                "Labor force participation rate": [10, 12, 14, 16, 18, 20],
                "GNI per capita": [100, 110, 130, 150, 170, 190],
                "Exports growth": [2, 1, 3, 2, 4, 3],
            }
        )
        out = analyze(df)
        self.assertIn("evidence", out)
        self.assertIn("statistic", out)
        self.assertGreater(out["statistic"], 0.8)

    def test_table_evidence_program_handles_wide_indicator_rows(self):
        import pandas as pd

        columns = {
            "Series Name": (
                "The name of the indicator or variable being measured: "
                "Adjusted savings: education expenditure - education spending. "
                "Labor force participation rate - human capital proxy. "
                "GNI per capita - economic output and income."
            ),
            "2000 [YR2000]": "value in 2000",
            "2001 [YR2001]": "value in 2001",
            "2002 [YR2002]": "value in 2002",
            "2003 [YR2003]": "value in 2003",
        }
        kernel = ResidualKernel()
        programs = kernel.induce_table_evidence_programs(
            question="How does increased education expenditure influence human capital and economic output?",
            column_descriptions=columns,
            domain_context="Labor force is a proxy for human capital. GNI per capita represents economic output.",
        )
        self.assertEqual(len(programs), 1)
        ns = {"pd": pd, "__builtins__": {"abs": abs, "dict": dict, "float": float, "len": len, "range": range, "round": round, "set": set, "str": str, "sum": sum}}
        exec(programs[0].code, ns)
        df = pd.DataFrame(
            {
                "Country Group": ["A", "A", "A"],
                "Series Name": [
                    "Adjusted savings: education expenditure (percentage of GNI)",
                    "Labor force participation rate",
                    "GNI per capita",
                ],
                "2000 [YR2000]": [1.0, 10.0, 100.0],
                "2001 [YR2001]": [2.0, 20.0, 200.0],
                "2002 [YR2002]": [3.0, 30.0, 300.0],
                "2003 [YR2003]": [4.0, 40.0, 400.0],
            }
        )
        out = ns["analyze"](df)
        self.assertIn("wide-mode", out["evidence"])
        self.assertGreater(out["statistic"], 0.95)

    def test_compresses_repeated_rewrite_rules(self):
        examples_a = [
            {"current": "", "target": "EEEEE", "context": {"main": "ABCDE", "vice": "EDCBA"}},
            {"current": "", "target": "EEEEE", "context": {"main": "EDCBA", "vice": "ABCDE"}},
        ]
        examples_b = [
            {"current": "", "target": "EDCDE", "context": {"main": "ABCDE", "vice": "EDCBA"}},
            {"current": "", "target": "EDCDE", "context": {"main": "EDCBA", "vice": "ABCDE"}},
        ]
        kernel = ResidualKernel()
        rules = [p.rule for p in kernel.induce_string_context_rules(examples_a)]
        rules.extend(p.rule for p in kernel.induce_string_context_rules(examples_b))
        compressed = compress_rewrite_rules(rules)
        self.assertLessEqual(len(compressed), len(rules))
        self.assertTrue(any(rule.support >= 2 for rule in compressed))


if __name__ == "__main__":
    unittest.main()
