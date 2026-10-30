"""Path resolution and access control for the PA_PB1 agent system.

The compartmentalisation invariant
----------------------------------
Agents may **read** anywhere under the repository root (datasets, PDBs, the
Julia packages, the existing ``src/`` pipeline) but may **write** only under
``agents/store/``.

Every filesystem path used by a tool must be produced by :func:`read_path` or
:func:`write_path`.  Tools never accept a raw path from the LLM: they accept
*identifiers* (a model name, a run id) which are validated here and mapped onto
a path.  An LLM cannot escape a sandbox it has no tool to address -- the guard
below is defence in depth, not the primary mechanism.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "AGENTS_ROOT",
    "REPO_ROOT",
    "STORE",
    "MODELS_DIR",
    "SEQUENCES_DIR",
    "DATASETS_DIR",
    "RUNS_DIR",
    "JULIA_DIR",
    "CONFIG_DIR",
    "WorkspaceError",
    "PathEscape",
    "InvalidName",
    "slug",
    "read_path",
    "write_path",
    "ensure_store",
    "run_dir",
    "model_dir",
]


# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------
# agents/core/workspace.py -> parents[1] == agents/ , parents[2] == PA_PB1/
AGENTS_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = AGENTS_ROOT.parent

#: The ONE writable root.  Nothing the agents produce may land outside it.
STORE: Path = AGENTS_ROOT / "store"

MODELS_DIR: Path = STORE / "models"
SEQUENCES_DIR: Path = STORE / "sequences"
DATASETS_DIR: Path = STORE / "datasets"
RUNS_DIR: Path = STORE / "runs"

#: Read-only code locations.
JULIA_DIR: Path = AGENTS_ROOT / "julia"
CONFIG_DIR: Path = AGENTS_ROOT / "config"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class WorkspaceError(RuntimeError):
    """Base class for workspace violations."""


class PathEscape(WorkspaceError, PermissionError):
    """A resolved path fell outside its permitted root."""


class InvalidName(WorkspaceError, ValueError):
    """An identifier supplied by an agent is not a safe path component."""


# --------------------------------------------------------------------------
# Identifier validation
# --------------------------------------------------------------------------
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Names that would be harmless as text but dangerous as path components.
_RESERVED = {".", "..", ""}


def slug(name: str) -> str:
    """Validate an agent-supplied identifier used as a single path component.

    Model names, run ids and dataset names all pass through here.  Anything
    containing a separator, a parent reference, or a leading dot is rejected
    outright rather than sanitised -- silent correction hides bugs and, with an
    LLM in the loop, hides probing.

    >>> slug("pnb-2r-3lay")
    'pnb-2r-3lay'
    >>> slug("../../etc/passwd")
    Traceback (most recent call last):
    ...
    agents.core.workspace.InvalidName: ...
    """
    if not isinstance(name, str):
        raise InvalidName(f"identifier must be a string, got {type(name).__name__}")
    if name in _RESERVED or "/" in name or "\\" in name:
        raise InvalidName(f"identifier is not a safe path component: {name!r}")
    if not _SLUG_RE.match(name):
        raise InvalidName(
            f"identifier {name!r} must match {_SLUG_RE.pattern} "
            "(letters, digits, dot, dash, underscore; max 64 chars)"
        )
    return name


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------
def _guard(path: Path, root: Path, what: str) -> Path:
    """Resolve ``path`` and assert it stays under ``root``.

    ``Path.resolve()`` normalises ``..`` *and* follows symlinks, so a symlink
    planted inside the store pointing at ``/etc`` is caught here too.
    """
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise PathEscape(f"{what} outside {root_resolved}: {resolved}")
    return resolved


def write_path(*parts: str | Path) -> Path:
    """Resolve a path for writing. Always under :data:`STORE`.

    Use for everything the agents produce: trained models, generated
    sequences, run traces, derived datasets.
    """
    return _guard(STORE.joinpath(*parts), STORE, "write")


def read_path(*parts: str | Path) -> Path:
    """Resolve a path for reading. Anywhere under :data:`REPO_ROOT`.

    Use for inputs the agents consume but must never modify: the affinity
    measurements, the sequencing counts, ``data/energy_model_params/``, the
    Julia packages, the existing ``src/`` scripts.
    """
    return _guard(REPO_ROOT.joinpath(*parts), REPO_ROOT, "read")


# --------------------------------------------------------------------------
# Conventional locations
# --------------------------------------------------------------------------
def ensure_store() -> None:
    """Create the writable tree if it does not exist yet. Idempotent."""
    for d in (MODELS_DIR, SEQUENCES_DIR, DATASETS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str, *, create: bool = False) -> Path:
    """Directory holding every artefact of a single orchestrator run."""
    p = write_path("runs", slug(run_id))
    if create:
        (p / "artifacts").mkdir(parents=True, exist_ok=True)
    return p


def model_dir(model_name: str, *, create: bool = False) -> Path:
    """Directory holding one trained model's weights and metadata."""
    p = write_path("models", slug(model_name))
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p
