import os

import pandas as pd
from unittest.mock import patch

from mars.induction.cpi_adapters import DiscoveryAdapter
from mars.induction.universal_cpi import UniversalCPI
from mars.skills.syndrome_decoder import decode_syndrome_program_sources


def test_syndrome_decoder_repairs_wide_table_axis_confusion():
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
    engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
    with patch.dict(os.environ, {"MARS_NOVA": "1"}, clear=False):
        nova_programs, nova_errors, _nova_props = engine._nova_programs(adapter, observations)
    assert not nova_errors

    decoded = decode_syndrome_program_sources(
        adapter=adapter,
        observations=observations,
        programs=nova_programs,
    )

    assert decoded is not None
    assert decoded.decoded_operator == "wide_table_rows_as_series"
    assert decoded.program_sources

    with patch.dict(os.environ, {"MARS_SYNDROME": "1"}, clear=False):
        syndrome_programs, syndrome_errors, _props = engine._syndrome_programs(
            adapter,
            observations,
            nova_programs,
        )
    assert not syndrome_errors
    assert syndrome_programs
    ranked = engine._rank(nova_programs + syndrome_programs, observations, adapter)
    assert ranked[0][0].name == "syndrome_wide_table_rows_as_series"
    report = adapter.render_report(ranked[:3], observations)
    assert "education expenditure" in report.lower()
    assert "human capital" in report.lower()
    assert "economic output" in report.lower()
