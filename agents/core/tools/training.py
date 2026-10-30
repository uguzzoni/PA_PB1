"""Training tools for the Trainer agent. Backed by ``PhageNegBinom.jl``.

Everything here crosses the Julia boundary via :mod:`agents.core.execution`,
so the same calls work unchanged when training moves to the cluster.

The one thing this module does *not* do is promote a model into
``data/energy_model_params/``.  That stays a human action -- see
``registry.promotion_instructions``.
"""

from __future__ import annotations

import time
from typing import Any

from .. import registry, workspace as ws
from ..execution import Runner, default_runner
from ..run import Run
from ..schemas import (
    ArchSpec,
    DatasetSpec,
    EvalMetrics,
    Hyperparams,
    ModelEntry,
    TrainConfig,
    TrainResult,
)

__all__ = ["prepare_dataset", "train_model", "evaluate_model", "describe_available_data"]


def describe_available_data() -> dict[str, Any]:
    """What measurement files exist, so the agent does not invent filenames.

    Read-only scan of the conventional data locations. Returns repo-relative
    paths suitable for :class:`DatasetSpec`.
    """
    out: dict[str, list[str]] = {"data": [], "energy_model": []}
    for key, rel in (("data", "data"), ("energy_model", "PD_energy_model/data")):
        root = ws.read_path(*rel.split("/"))
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".tsv", ".json"}:
                out[key].append(str(p.relative_to(ws.REPO_ROOT)))
    return out


def prepare_dataset(spec: DatasetSpec) -> dict[str, Any]:
    """Validate a dataset specification and report what it points at.

    Cheap pre-flight so a bad path fails here, in seconds, rather than after
    Julia has spun up.
    """
    info: dict[str, Any] = {"ok": True, "problems": []}
    for field in ("counts_file", "affinity_file"):
        rel = getattr(spec, field)
        if rel is None:
            continue
        p = ws.read_path(rel)
        if not p.exists():
            info["ok"] = False
            info["problems"].append(f"{field}: no such file: {rel}")
        else:
            info[field] = {
                "path": str(p.relative_to(ws.REPO_ROOT)),
                "size_bytes": p.stat().st_size,
                "n_lines": sum(1 for _ in p.open(encoding="utf-8", errors="replace")),
            }
    info["val_fraction"] = spec.val_fraction
    return info


def train_model(
    arch: ArchSpec,
    dataset: DatasetSpec,
    model_name: str,
    hyperparams: Hyperparams | None = None,
    *,
    run: Run | None = None,
    runner: Runner | None = None,
    overwrite: bool = False,
) -> TrainResult:
    """Train an energy model and register it.

    The trained weights land in ``agents/store/models/<model_name>/`` and the
    model is added to the registry. Nothing is written outside the store, and
    the ColabDesign pipeline is unaffected until you promote it by hand.
    """
    ws.slug(model_name)
    if model_name in registry.load_registry() and not overwrite:
        raise FileExistsError(
            f"model {model_name!r} already exists. Pick a different name so the "
            "comparison history stays intact, or pass overwrite=True."
        )

    pre = prepare_dataset(dataset)
    if not pre["ok"]:
        raise ValueError(f"dataset is not usable: {pre['problems']}")

    config = TrainConfig(
        model_name=model_name,
        arch=arch,
        dataset=dataset,
        hyperparams=hyperparams or Hyperparams(),
    )

    own_run = run is None
    run = run or Run.create(f"train-{model_name}", goal=f"train {model_name}")
    if own_run:
        run.__enter__()

    started = time.time()
    try:
        runner = runner or default_runner()
        ws.model_dir(model_name, create=True)
        result = runner.run("train.jl", config, TrainResult, run_dir=run.dir, tag=model_name)
        run.trace("train_model", config.model_dump(), result.model_dump())
    except Exception as exc:
        run.trace("train_model", config.model_dump(), None, error=str(exc))
        raise
    finally:
        if own_run:
            run.__exit__(None, None, None)

    registry.register_model(
        ModelEntry(
            name=model_name,
            arch=arch,
            metrics=result.metrics,
            weights_path=result.weights_path,
            n_params=result.n_params or arch.n_params(),
            run_id=run.run_id,
        ),
        overwrite=overwrite,
    )
    result.duration_s = round(time.time() - started, 2)
    return result


def evaluate_model(
    model_name: str,
    sequences: list[str],
    measured: list[float],
    *,
    objective: str = "minimize",
) -> EvalMetrics:
    """Correlate predicted energies against measured values. Pure Python, no Julia.

    This is the zero-cost validation to run first: score the sequences in
    ``data/seqs_aff_alberto_darren*.txt`` against
    ``data/affinity_measurements_PA-PB1_formatted.csv`` and check the rank
    correlation is what you expect. It validates the weight loader, the
    alphabet ordering and the sign convention in one shot.
    """
    if len(sequences) != len(measured):
        raise ValueError(f"{len(sequences)} sequences vs {len(measured)} measurements")
    if len(sequences) < 3:
        raise ValueError("need at least 3 points to correlate")

    from .fitness import predict_fitness  # local import: keeps numpy off the import path

    preds = [s.energy for s in predict_fitness(sequences, model_name)]
    return EvalMetrics(
        spearman=_spearman(preds, measured),
        pearson=_pearson(preds, measured),
        n=len(sequences),
    )


# -- small stats helpers, so the tool layer needs no scipy -----------------
def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    den = (sum(x * x for x in da) ** 0.5) * (sum(y * y for y in db) ** 0.5)
    return round(num / den, 4) if den else None


def _spearman(a: list[float], b: list[float]) -> float | None:
    return _pearson(_rank(a), _rank(b))
