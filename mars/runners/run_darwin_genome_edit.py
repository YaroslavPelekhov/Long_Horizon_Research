"""Edit a regulatory genome for causal ablations.

Examples:
  python -m mars.runners.run_darwin_genome_edit \\
    --input_genome lmw/darwin_outer/g3/regulatory_genome.json \\
    --remove family_table_robust_mediation_developer \\
    --output_dir lmw/darwin_outer/g3_no_family
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from mars.darwin import RegulatoryGenome  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_genome", required=True)
    parser.add_argument("--remove", default="")
    parser.add_argument("--priority_gene", default="")
    parser.add_argument("--priority", type=float, default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    genome = RegulatoryGenome.load_json(args.input_genome)
    removed = [x.strip() for x in args.remove.split(",") if x.strip()]
    edited = genome
    if removed:
        edited = edited.without_genes(removed)
    if args.priority_gene:
        if args.priority is None:
            raise SystemExit("--priority is required with --priority_gene")
        edited = edited.with_gene_priority(args.priority_gene, args.priority)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _PROJ / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    genome_path = out_dir / "regulatory_genome.json"
    manifest_path = out_dir / "edit_manifest.json"
    edited.save_json(genome_path)
    manifest_path.write_text(
        json.dumps(
            {
                "input_genome": str(Path(args.input_genome).resolve()),
                "output_genome": str(genome_path),
                "removed": removed,
                "priority_gene": args.priority_gene,
                "priority": args.priority,
                "n_input_genes": len(genome.genes),
                "n_output_genes": len(edited.genes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"genome → {genome_path}")
    print(f"manifest → {manifest_path}")


if __name__ == "__main__":
    main()
