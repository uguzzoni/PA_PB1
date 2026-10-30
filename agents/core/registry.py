"""The model registry: ``agents/store/models/registry.json``.

Why a registry and not a directory glob: it is the Architect agent's single
source of truth about what exists, and it answers ``list_trained_models``
without giving any agent a filesystem-listing tool.  It is also the second
control point on what the agents have produced -- you can read one file to see
the whole history.

Promotion to ``data/energy_model_params/`` (the weights the ColabDesign
pipeline actually loads) is **not** implemented here, on purpose.  It is a
human action.  :func:`promotion_instructions` prints what to do.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .schemas import ModelEntry
from . import workspace as ws

__all__ = ["registry_path", "load_registry", "register_model", "get_model", "list_models",
           "promotion_instructions"]


def registry_path() -> Path:
    return ws.write_path("models", "registry.json")


def load_registry() -> dict[str, ModelEntry]:
    """Read the registry. Returns an empty mapping if it does not exist yet."""
    p = registry_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {name: ModelEntry.model_validate(entry) for name, entry in raw.items()}


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then rename.

    A half-written registry after a crash would be worse than a lost run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def register_model(entry: ModelEntry, *, overwrite: bool = False) -> ModelEntry:
    """Add a trained model to the registry.

    Raises if the name already exists and ``overwrite`` is False -- an agent
    silently replacing a model you were comparing against is a bug you would
    only notice much later.
    """
    ws.slug(entry.name)
    reg = load_registry()
    if entry.name in reg and not overwrite:
        raise FileExistsError(
            f"model {entry.name!r} already registered (created {reg[entry.name].created}). "
            "Choose another name or pass overwrite=True explicitly."
        )
    reg[entry.name] = entry
    _atomic_write(
        registry_path(),
        json.dumps({k: v.model_dump() for k, v in sorted(reg.items())}, indent=2),
    )
    return entry


def get_model(name: str) -> ModelEntry:
    reg = load_registry()
    key = ws.slug(name)
    if key not in reg:
        known = ", ".join(sorted(reg)) or "(none)"
        raise KeyError(f"no model named {key!r}. Registered models: {known}")
    return reg[key]


def list_models() -> list[ModelEntry]:
    return sorted(load_registry().values(), key=lambda e: e.created, reverse=True)


def promotion_instructions(name: str) -> str:
    """What a human runs to put an agent-trained model into production."""
    entry = get_model(name)
    src = ws.write_path(entry.weights_path)
    dst = ws.REPO_ROOT / "data" / "energy_model_params" / Path(entry.weights_path).name
    return (
        f"Promotion is a human decision -- no tool does this.\n"
        f"  cp {src} {dst}\n"
        f"Then point ENERGY_WEIGHTS_PATH at it and re-run the ColabDesign protocols.\n"
        f"Model {entry.name!r}: {entry.n_params} params, metrics={entry.metrics.model_dump()}"
    )
