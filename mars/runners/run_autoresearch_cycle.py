"""Run the workspace-level autonomous research cycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv

    env_path = _PROJ / "autodiscovery" / ".env.local"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass

from mars.autoresearch import ResearchCycle  # noqa: E402
from mars.autoresearch.core import render_markdown  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="autoresearch_cycle_v1")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _PROJ / "lmw" / "autoresearch" / args.run_id
    state_path = out_dir / "state.json"
    memo_path = out_dir / "research_plan.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and (state_path.exists() or memo_path.exists()):
        raise SystemExit(f"output exists; pass --overwrite: {out_dir}")

    state = ResearchCycle(_PROJ).run()
    state_path.write_text(json.dumps(state.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")
    memo_path.write_text(render_markdown(state), encoding="utf-8")

    print("=== MARS autonomous research cycle ===")
    print(f"state={state_path}")
    print(f"memo={memo_path}")
    if state.selected:
        print(f"selected={state.selected.name} priority={state.selected.priority:.2f}")
        print(f"next={state.selected.next_experiment}")


if __name__ == "__main__":
    main()
