"""Analysis tools: diversity, ranking, filtering, persistence of sequence sets.

The diversity metrics exist for a specific failure mode.  An MCMC chain guided
by a fitness model collapses onto a single motif, and an LLM shown only the
mean energy will report that as success.  Giving the Generator agent
:func:`diversity_metrics` as a first-class tool -- and telling it in the system
prompt to check both -- is what prevents a confident write-up of twenty copies
of the same peptide.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .. import workspace as ws
from ..schemas import AA_ALPHABET, DiversityMetrics, SeqConstraints, SeqRecord

__all__ = [
    "hamming",
    "diversity_metrics",
    "rank_sequences",
    "filter_sequences",
    "save_sequences",
    "load_sequences",
]


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b) if x != y)


def diversity_metrics(
    sequences: Sequence[str], *, identity_threshold: float = 0.8, max_pairs: int = 500
) -> DiversityMetrics:
    """Summarise how varied a sequence set is.

    Pairwise distances are computed on at most ``max_pairs`` sequences; beyond
    that the estimate is taken on a prefix, which is enough to detect collapse
    and keeps the tool fast enough for an agent loop.
    """
    seqs = list(sequences)
    if not seqs:
        return DiversityMetrics(
            n=0, mean_hamming=0.0, min_hamming=0, n_unique=0, n_clusters=0,
            mean_position_entropy=0.0,
        )

    L = len(seqs[0])
    sample = seqs[:max_pairs]

    dists = [hamming(sample[i], sample[j])
             for i in range(len(sample)) for j in range(i + 1, len(sample))]
    mean_h = sum(dists) / len(dists) if dists else 0.0
    min_h = min(dists) if dists else 0

    # Greedy single-linkage clustering at the identity threshold.
    cutoff = int(round((1.0 - identity_threshold) * L))
    centroids: list[str] = []
    for s in sample:
        if all(hamming(s, c) > cutoff for c in centroids):
            centroids.append(s)

    # Mean per-position Shannon entropy, in bits.
    entropies = []
    for pos in range(L):
        counts = Counter(s[pos] for s in seqs)
        total = sum(counts.values())
        entropies.append(-sum((c / total) * math.log2(c / total) for c in counts.values()))

    return DiversityMetrics(
        n=len(seqs),
        mean_hamming=round(mean_h, 3),
        min_hamming=min_h,
        n_unique=len(set(seqs)),
        n_clusters=len(centroids),
        mean_position_entropy=round(sum(entropies) / L, 3),
    )


def rank_sequences(
    records: Sequence[SeqRecord], *, top_k: int | None = None, objective: str = "minimize"
) -> list[SeqRecord]:
    """Sort by energy. Lower is better unless ``objective='maximize'``."""
    ordered = sorted(records, key=lambda r: r.energy, reverse=(objective == "maximize"))
    return ordered[:top_k] if top_k else ordered


def filter_sequences(
    records: Sequence[SeqRecord],
    constraints: SeqConstraints,
    *,
    seeds: Sequence[str] = (),
    drop_duplicates: bool = True,
) -> list[SeqRecord]:
    """Apply hard constraints. Returns only the records that pass."""
    forbidden = set(constraints.forbidden_aa.upper())
    seen: set[str] = set()
    kept: list[SeqRecord] = []

    for rec in records:
        if drop_duplicates and rec.seq in seen:
            continue
        if forbidden & set(rec.seq):
            continue
        if any(rec.seq[p] != aa.upper() for p, aa in constraints.fixed_positions.items()):
            continue
        if constraints.max_identity_to_seeds is not None and seeds:
            L = len(rec.seq)
            worst = max(
                (L - hamming(rec.seq, s)) / L for s in seeds if len(s) == L
            ) if any(len(s) == L for s in seeds) else 0.0
            if worst > constraints.max_identity_to_seeds:
                continue
        seen.add(rec.seq)
        kept.append(rec)
    return kept


# --------------------------------------------------------------------------
# Persistence (always inside the store)
# --------------------------------------------------------------------------
def save_sequences(records: Sequence[SeqRecord], name: str, *, fmt: str = "both") -> dict[str, str]:
    """Write a sequence set to ``agents/store/sequences/<name>.{json,csv}``."""
    stem = ws.slug(name)
    out: dict[str, str] = {}
    ws.SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

    if fmt in ("json", "both"):
        p = ws.write_path("sequences", f"{stem}.json")
        p.write_text(json.dumps([r.model_dump() for r in records], indent=2), encoding="utf-8")
        out["json"] = str(p)
    if fmt in ("csv", "both"):
        p = ws.write_path("sequences", f"{stem}.csv")
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["seq", "energy", "source"])
            for r in records:
                w.writerow([r.seq, r.energy, r.source])
        out["csv"] = str(p)
    return out


def load_sequences(name: str) -> list[SeqRecord]:
    p = ws.write_path("sequences", f"{ws.slug(name)}.json")
    if not p.exists():
        raise FileNotFoundError(f"no saved sequence set named {name!r} ({p})")
    return [SeqRecord.model_validate(d) for d in json.loads(p.read_text(encoding="utf-8"))]


def read_seed_sequences(repo_relative_path: str) -> list[str]:
    """Read seed/measured sequences from the read-only side of the repo.

    e.g. ``data/input_sequences.txt`` or ``data/seqs_aff_alberto_darren.txt``.
    One sequence per line; blank lines and ``#`` comments ignored.
    """
    p = ws.read_path(repo_relative_path)
    if not p.exists():
        raise FileNotFoundError(f"no such file in the repository: {p}")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip().split()[0] if line.strip() else ""
        if line and not line.startswith("#"):
            out.append(line.upper())
    return out
