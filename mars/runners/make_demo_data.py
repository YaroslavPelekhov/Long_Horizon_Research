"""Generate real trace data for the compression-amplification demo.

For each showcase task we record everything the HTML demo needs to animate:
  - the observations (input -> output pairs)
  - strong model (gpt-4o) single-shot answer + whether it fits
  - weak model (gpt-4o-mini) single-shot answer + whether it fits
  - weak+MDL full search trace: every candidate per round (code, bits, exact,
    accepted), the winning program, and library growth.

Output: demo/compression_demo_data.json  (consumed by the standalone HTML demo)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "ultrahorizon_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.mdl.engine import MDLEngine, Library, make_llm_proposer, _compile
from mars.mdl.tasks_grid import GridLetterTask
from mars.runners.run_mdl_grid import collect as grid_collect, _RAW as GRID_RAW, _obs_json as grid_obs_json, _extract as grid_extract, _eval_code as grid_eval

WEAK = "openai/gpt-4o-mini"
STRONG = "openai/gpt-4o"


def grid_task_payload(letter: str, obs, engine: MDLEngine, client) -> dict:
    # raw baselines
    def raw(model):
        code = grid_extract(call_llm(client, model=model,
            system="You write Python functions that reproduce hidden grid-tile effects.",
            user=GRID_RAW.format(obs=grid_obs_json(obs)), max_tokens=400, temperature=0.2))
        return code, grid_eval(code, obs)
    weak_code, weak_fit = raw(WEAK)
    strong_code, strong_fit = raw(STRONG)
    # weak + MDL
    res = engine.compress(GridLetterTask(letter, obs))
    trace = getattr(res, "trace", [])
    return {
        "kind": "grid", "letter": letter,
        "observations": [{"x": o.x, "y": o.y, "energy": o.energy, "steps": o.steps,
                          "visit_count": o.visit_count, "delta_score": o.delta_score} for o in obs],
        "weak_raw": {"code": weak_code, "exact": round(weak_fit, 3), "solved": weak_fit >= 0.999},
        "strong_raw": {"code": strong_code, "exact": round(strong_fit, 3), "solved": strong_fit >= 0.999},
        "weak_mdl": {
            "winner_code": res.best.code if res.best else None,
            "exact": round(res.exact_rate, 3), "solved": res.exact_rate >= 0.999,
            "rounds": res.rounds, "n_proposed": res.n_proposed, "n_valid": res.n_valid,
            "trace": trace,
        },
        "library_after": engine.library.names(),
    }


def main():
    client = make_openai_client()
    out = {"showcase": [], "principle": "shortest program reproducing observations; bits judge, not the model",
           "weak": WEAK, "strong": STRONG}

    # One shared engine+library across letters → show transfer (library growth)
    engine = MDLEngine(make_llm_proposer(WEAK), max_rounds=4, k_per_round=10, library=Library())

    obs_by_letter, _ = grid_collect(3, "easy")
    # showcase letters: C (position parity) and E (energy threshold) are the
    # nontrivial ones where strong often fails and the search is interesting.
    for L in ["C", "E", "A"]:
        ob = obs_by_letter.get(L, [])
        if len(ob) < 4:
            continue
        print(f"generating showcase for letter {L} ({len(ob)} obs)...", flush=True)
        payload = grid_task_payload(L, ob, engine, client)
        out["showcase"].append(payload)
        print(f"  weak_raw solved={payload['weak_raw']['solved']} "
              f"strong_raw solved={payload['strong_raw']['solved']} "
              f"weak+MDL solved={payload['weak_mdl']['solved']} "
              f"(lib now {len(payload['library_after'])})", flush=True)

    demo_dir = _PROJ / "demo"
    demo_dir.mkdir(exist_ok=True)
    out_path = demo_dir / "compression_demo_data.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}  ({len(out['showcase'])} showcases)")


if __name__ == "__main__":
    main()
