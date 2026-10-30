"""Tool layer: plain Python functions, no LLM, no framework.

Every function here is independently testable and independently usable from a
notebook. That is deliberate -- when something goes wrong in an agent run, you
want to reproduce it without the agent.
"""

from . import analysis, architecture, fitness, training  # noqa: F401

__all__ = ["analysis", "architecture", "fitness", "training"]
