"""Architecture tools for the Architect agent.

Design note: choosing an architecture is the LLM's job, not a tool's.  These
tools exist only to give it ground truth -- what already exists, how it is
built, how it performed, and whether a proposed design is buildable at all.
Without ``validate_architecture`` and ``estimate_params`` an LLM will cheerfully
propose a model with 40M parameters trained on a few thousand measurements, or
one whose output layer cannot emit a scalar energy.
"""

from __future__ import annotations

import json
from typing import Any

from .. import registry, workspace as ws
from ..execution import Runner, default_runner
from ..schemas import ArchSpec, ModelEntry

__all__ = [
    "list_trained_models",
    "inspect_model",
    "validate_architecture",
    "estimate_params",
    "compare_models",
    "list_reference_models",
]


def list_trained_models() -> list[dict[str, Any]]:
    """Every model the agents have trained, newest first.

    Returns a compact view -- name, architecture summary, metrics -- rather
    than full weight paths, so the LLM's context is not filled with noise.
    """
    out = []
    for e in registry.list_models():
        out.append(
            {
                "name": e.name,
                "created": e.created,
                "n_params": e.n_params,
                "n_layers": len(e.arch.layers),
                "hidden_units": [l.units for l in e.arch.layers[:-1]],
                "n_rounds": e.arch.n_rounds,
                "likelihood": e.arch.likelihood,
                "metrics": e.metrics.model_dump(exclude_none=True),
                "promoted": e.promoted,
                "notes": e.arch.notes,
            }
        )
    return out


def list_reference_models() -> list[dict[str, Any]]:
    """Pre-existing weights in ``data/energy_model_params/`` (read-only).

    These are the models the ColabDesign pipeline currently uses. The Architect
    agent should look here before proposing anything, so it designs *relative
    to* what already works rather than from scratch.
    """
    d = ws.read_path("data", "energy_model_params")
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            keys = list(raw) if isinstance(raw, dict) else []
        except (json.JSONDecodeError, OSError):
            keys = []
        out.append({"file": p.name, "top_level_keys": keys[:20], "size_bytes": p.stat().st_size})
    return out


def inspect_model(model_name: str, *, runner: Runner | None = None, use_julia: bool = False) -> dict[str, Any]:
    """Full detail on one registered model: architecture, metrics, provenance."""
    entry = registry.get_model(model_name)
    detail: dict[str, Any] = {
        "name": entry.name,
        "arch": entry.arch.model_dump(),
        "metrics": entry.metrics.model_dump(exclude_none=True),
        "n_params": entry.n_params,
        "weights_path": entry.weights_path,
        "run_id": entry.run_id,
        "created": entry.created,
        "promoted": entry.promoted,
    }
    if use_julia:
        run_dir = ws.run_dir("adhoc-inspect", create=True)
        runner = runner or default_runner()
        detail["julia"] = runner.run(
            "inspect.jl", entry.arch, ArchSpec, run_dir=run_dir, tag=entry.name
        ).model_dump()
    return detail


def validate_architecture(arch: ArchSpec, *, n_training_examples: int | None = None) -> dict[str, Any]:
    """Check a proposed architecture is buildable and not obviously over-sized.

    Pydantic already enforces the structural rules (scalar identity output).
    This adds the judgement calls an LLM tends to get wrong, as warnings rather
    than hard failures -- the agent should see them and decide.
    """
    warnings: list[str] = []
    n_params = arch.n_params()

    if n_training_examples:
        ratio = n_params / max(n_training_examples, 1)
        if ratio > 10:
            warnings.append(
                f"{n_params} parameters for {n_training_examples} training examples "
                f"({ratio:.1f}x). Severe overfitting risk; shrink the hidden layers."
            )
        elif ratio > 1:
            warnings.append(
                f"{n_params} parameters vs {n_training_examples} examples ({ratio:.1f}x). "
                "Regularisation will be load-bearing."
            )

    hidden = [l.units for l in arch.layers[:-1]]
    if hidden and any(h > arch.input_dim for h in hidden):
        warnings.append(
            f"hidden layer wider than the input ({max(hidden)} > {arch.input_dim}); "
            "rarely helps for one-hot sequence input."
        )
    if len(arch.layers) > 5:
        warnings.append(f"{len(arch.layers)} layers is deep for this data regime.")
    if not arch.regularization.get("l2"):
        warnings.append("no L2 regularisation set.")

    return {
        "valid": True,          # structural validity is guaranteed by pydantic
        "n_params": n_params,
        "input_dim": arch.input_dim,
        "layer_sizes": [l.units for l in arch.layers],
        "warnings": warnings,
    }


def estimate_params(arch: ArchSpec) -> int:
    """Trainable parameter count, biases included."""
    return arch.n_params()


def compare_models(names: list[str]) -> dict[str, Any]:
    """Side-by-side comparison of registered models.

    ``best_by`` is reported per metric so the agent cannot collapse a
    multi-criteria comparison into a single winner without saying on what.
    """
    entries: list[ModelEntry] = [registry.get_model(n) for n in names]
    rows = [
        {
            "name": e.name,
            "n_params": e.n_params,
            "layers": [l.units for l in e.arch.layers],
            "n_rounds": e.arch.n_rounds,
            **e.metrics.model_dump(exclude_none=True),
        }
        for e in entries
    ]

    best: dict[str, str] = {}
    for metric, better_is_higher in (("spearman", True), ("pearson", True), ("val_loss", False)):
        vals = [(r["name"], r.get(metric)) for r in rows if r.get(metric) is not None]
        if vals:
            best[metric] = (max if better_is_higher else min)(vals, key=lambda kv: kv[1])[0]

    return {"models": rows, "best_by": best}
