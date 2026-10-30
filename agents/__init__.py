"""PA_PB1 agent system.

A multi-agent layer over the existing binder-design pipeline. Phase 1 covers
the fitness (energy) model only: architecture, training, and sequence
generation, all on CPU. ColabDesign and cluster execution arrive in phase 2
behind the Runner interface already defined in ``core/execution.py``.

Compartmentalisation: agents read anywhere in the repository and write only
under ``agents/store/``. See ``core/workspace.py``.
"""

__all__ = ["core"]
__version__ = "0.1.0"
