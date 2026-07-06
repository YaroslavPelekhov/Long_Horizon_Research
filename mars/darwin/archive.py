"""Quality-diversity archive for MARS theory genomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mars.darwin.theory_genome import TheoryGenome


@dataclass(frozen=True)
class NicheKey:
    """Behavioral niche for a theory genome."""

    object_family: str
    gene_family: str
    relation_family: str
    evidence_family: str
    complexity_band: str


@dataclass(frozen=True)
class EliteRecord:
    """Best genome found for a niche."""

    niche: NicheKey
    genome: TheoryGenome
    fitness: float
    metadata: Mapping[str, Any]


class TheoryArchive:
    """Minimal MAP-Elites-style archive for theory genomes."""

    def __init__(self) -> None:
        self._records: dict[NicheKey, EliteRecord] = {}

    def add(
        self,
        genome: TheoryGenome,
        *,
        fitness: float,
        niche: NicheKey | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        key = niche or niche_for_genome(genome)
        current = self._records.get(key)
        if current is not None and current.fitness >= fitness:
            return False
        self._records[key] = EliteRecord(
            niche=key,
            genome=genome,
            fitness=float(fitness),
            metadata=dict(metadata or {}),
        )
        return True

    def elites(self) -> tuple[EliteRecord, ...]:
        return tuple(sorted(self._records.values(), key=lambda r: r.fitness, reverse=True))

    def to_trace(self) -> list[dict[str, Any]]:
        rows = []
        for record in self.elites():
            rows.append(
                {
                    "niche": record.niche.__dict__,
                    "genome": record.genome.name,
                    "kind": record.genome.kind,
                    "fitness": record.fitness,
                    "metadata": dict(record.metadata),
                }
            )
        return rows


def niche_for_genome(genome: TheoryGenome) -> NicheKey:
    genes = dict(genome.genes)
    orientation = str(genes.get("orientation") or genes.get("input_family") or "generic")
    relation = str(genes.get("relation") or genes.get("relation_family") or genome.kind)
    evidence = str(genes.get("evidence") or genes.get("evidence_family") or "executable")
    if genome.complexity < 2:
        band = "small"
    elif genome.complexity < 4:
        band = "medium"
    else:
        band = "large"
    return NicheKey(
        object_family=orientation,
        gene_family=genome.kind,
        relation_family=relation,
        evidence_family=evidence,
        complexity_band=band,
    )
