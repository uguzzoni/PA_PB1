"""Core of the PA_PB1 agent system.

Layering, strictly downward:

    definitions/   LLM agents: system prompts + the tools each may call
    tools/         plain Python functions (the only things that touch data)
    execution.py   the Julia / SLURM boundary
    registry.py    what models exist
    run.py         run directories and the audit trace
    workspace.py   path resolution and the write sandbox
    schemas.py     pydantic contracts shared by all of the above

Nothing below ``definitions/`` imports slapsci, so the whole system is
testable without a provider or an API key.
"""

from . import execution, registry, run, schemas, workspace  # noqa: F401

__all__ = ["execution", "registry", "run", "schemas", "workspace"]
__version__ = "0.1.0"
