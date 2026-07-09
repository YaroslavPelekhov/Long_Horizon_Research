import os

import pandas as pd
from unittest.mock import patch

from mars.darwin import DarwinSynthesizer
from mars.induction.cpi_adapters import DiscoveryAdapter
from mars.induction.universal_cpi import UniversalCPI


def test_darwin_evolves_wide_table_rows_as_variables_genome():
    df = pd.DataFrame(
        {
            "Country Group": ["X", "X", "X"],
            "Series Name": [
                "Adjusted savings: education expenditure",
                "Labor force participation rate",
                "GNI per capita",
            ],
            "1990 [YR1990]": [2.0, 50.0, 100.0],
            "1991 [YR1991]": [3.0, 55.0, 130.0],
            "1992 [YR1992]": [4.0, 60.0, 160.0],
            "1993 [YR1993]": [5.0, 65.0, 190.0],
            "1994 [YR1994]": [6.0, 70.0, 220.0],
        }
    )
    adapter = DiscoveryAdapter(
        df,
        "How does increased education expenditure influence human capital and economic output?",
        "Human capital can be proxied by labor force; economic output can be proxied by GNI per capita.",
        {"Series Name": "indicator or measured variable"},
        enabled_modules=set(),
    )
    observations = adapter.collect_observations()
    result = DarwinSynthesizer(max_programs=8).synthesize(
        signature_hint=adapter.signature_hint(),
        observations=observations,
        interface_name=adapter.name,
        question=adapter.question,
        column_descriptions=adapter.column_descriptions,
        domain_context=adapter.domain_knowledge,
    )
    names = [src["name"] for src in result.program_sources]
    assert "darwin_rows_as_variables_chain" in names

    engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
    with patch.dict("os.environ", {"MARS_DARWIN": "1"}, clear=False):
        programs, errors, proposals = engine._darwin_programs(adapter, observations)
    assert not errors
    assert any(p.get("kind") == "darwin" for p in proposals)
    ranked = engine._rank(programs, observations, adapter)
    assert ranked[0][0].name.startswith("darwin_rows_as_variables")
    report = adapter.render_report(ranked[:3], observations)
    assert "education expenditure" in report.lower()
    assert "human capital" in report.lower()
    assert "economic output" in report.lower()
