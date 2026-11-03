"""Runtime configuration for the agent system, backed by ``agents/config/default.yaml``.

The file is read **at import time** by :mod:`agents.core.execution` and by the
agent definitions in :mod:`agents.core.definitions`, so the values below become
the defaults baked into the tool JSON schemas the LLM is shown.  Resolving them
inside function bodies instead would erase the defaults from those schemas,
which is worse than the small indirection of reading a module-level constant.

Three properties worth keeping:

* **It degrades.**  :data:`DEFAULTS` is a literal mirror of the shipped YAML and
  is always the merge base, so a missing, partial or malformed file leaves the
  system working with the values it had before.
* **It is read-only to agents.**  No tool exposes this module, and nothing here
  writes.  Editing configuration is a human action, like promotion.
* **It is overridable.**  Point ``PA_PB1_AGENTS_CONFIG`` at another file to run
  an experiment without touching the checked-in one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import workspace as ws

__all__ = ["DEFAULTS", "CONFIG_ENV_VAR", "config_path", "load_config", "get"]

#: Environment variable overriding which file :func:`load_config` reads.
CONFIG_ENV_VAR = "PA_PB1_AGENTS_CONFIG"

#: Fallback values, mirroring ``agents/config/default.yaml``.
#:
#: Kept in sync by ``tests/test_config.py``, which asserts the shipped file
#: introduces no key this mapping does not already describe.
DEFAULTS: dict[str, Any] = {
    "llm": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "per_agent": {
            "architect": None,
            "trainer": None,
            "generator": None,
            "orchestrator": None,
        },
    },
    "execution": {
        "julia_project": "PD_energy_model",
        "julia_bin": "julia",
        "timeout_s": 3600,
        "backend": "local",
    },
    "defaults": {
        "seq_length": 15,
        "forbidden_aa": "C",
        "objective": "minimize",
    },
    "sampling": {
        "n": 50,
        "temperature": 1.0,
        "n_steps": 1000,
        "min_mean_hamming": 3.0,
    },
    "data": {
        "seed_sequences": "data/input_sequences.txt",
        "affinity_measurements": "data/affinity_measurements_PA-PB1_formatted.csv",
        "reference_weights_dir": "data/energy_model_params",
    },
}

_cache: dict[str, Any] | None = None


def config_path() -> Path:
    """The file :func:`load_config` reads, honouring :data:`CONFIG_ENV_VAR`."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return ws.CONFIG_DIR / "default.yaml"


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """``over`` wins, except that nested mappings merge rather than replace.

    So a file setting only ``sampling.n`` keeps the other ``sampling`` keys.
    """
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, *, reload: bool = False) -> dict[str, Any]:
    """Return the merged configuration. Cached; pass ``reload=True`` to re-read.

    A missing file is not an error -- :data:`DEFAULTS` is a complete
    configuration on its own.  A file that exists but cannot be parsed *is*
    reported, as a warning rather than an exception: a typo in a config file
    should not take down a training run, but it must not pass unnoticed either.
    """
    global _cache
    if path is None and _cache is not None and not reload:
        return _cache

    target = Path(path).expanduser() if path is not None else config_path()
    merged = DEFAULTS
    if target.is_file():
        try:
            import yaml

            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        except ImportError:  # pragma: no cover - PyYAML is a declared dependency
            import warnings

            warnings.warn(
                f"PyYAML is not installed, so {target} was ignored and built-in "
                "defaults are in use. Install it with `uv pip install pyyaml`.",
                RuntimeWarning,
                stacklevel=2,
            )
            loaded = None
        except Exception as exc:  # noqa: BLE001 - any parse failure, reported not raised
            import warnings

            warnings.warn(
                f"could not parse {target} ({exc}); falling back to built-in defaults",
                RuntimeWarning,
                stacklevel=2,
            )
            loaded = None
        if isinstance(loaded, dict):
            merged = _deep_merge(DEFAULTS, loaded)

    if path is None:
        _cache = merged
    return merged


def get(dotted_key: str, default: Any = None) -> Any:
    """Look a value up by dotted path, e.g. ``get("sampling.n_steps")``.

    Returns ``default`` if any segment is missing, so a caller never has to
    guard against a partially written configuration file.
    """
    node: Any = load_config()
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
