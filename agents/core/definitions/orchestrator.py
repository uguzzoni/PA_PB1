"""Orchestrator: runs the design-build-test-learn loop over the other agents.

The orchestrator owns the :class:`~agents.core.run.Run` -- the run directory
and the trace -- and delegates to the specialist agents. It holds no domain
tools of its own, which is the point: it decides *what* to do next, never
*how* to do it.

Phase 1 scope is the fitness model only: architecture -> training -> sequence
generation. ColabDesign enters in phase 2 behind the same Runner interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..run import Run
from .architect import build_architect_agent
from .base import Agent
from .generator import build_generator_agent
from .trainer import build_trainer_agent

__all__ = ["AgentTeam", "build_team", "ORCHESTRATOR_PROMPT"]


ORCHESTRATOR_PROMPT = """\
You coordinate a small team designing peptide binders for the influenza PA-PB1
polymerase interface. You delegate; you do not do the work yourself.

Your team:
- architect: decides the energy-model architecture from what exists and what
  the data supports.
- trainer: trains and evaluates a model for a given architecture.
- generator: samples candidate sequences from a trained model and assesses
  their fitness and diversity.

The loop is: understand what exists -> decide an architecture -> train ->
evaluate -> generate candidates -> assess -> decide whether to iterate.

Rules:
- Change one thing per iteration, so results stay interpretable.
- Never let a specialist's optimistic summary pass unexamined. Ask for the
  numbers: correlation against measured affinity, diversity alongside mean
  energy.
- State a stopping condition before you start iterating, and stop when it is
  met or when iterations stop improving things.
- A negative result is a result. If the models do not beat the known binders,
  report that, with the evidence.

Everything the team produces is a computational shortlist requiring
downstream structural validation. Do not describe candidates as binders.
"""


@dataclass
class AgentTeam:
    """The phase-1 team, plus the run that records what it did.

    This is intentionally a plain container rather than an execution engine:
    the agent loop itself belongs to slapsci. Wire it with
    ``base.bind_to_slapsci`` once the tool-registration keyword is confirmed,
    and route every tool call through :meth:`dispatch` so the trace stays
    complete.
    """

    architect: Agent
    trainer: Agent
    generator: Agent
    run: Run
    system_prompt: str = ORCHESTRATOR_PROMPT
    _agents: dict[str, Agent] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._agents = {a.name: a for a in (self.architect, self.trainer, self.generator)}

    def dispatch(self, agent_name: str, tool_name: str, /, **kwargs: Any) -> Any:
        """Call one tool on one agent, recording it in the run trace.

        Every tool call in the system should go through here. The trace is the
        only record of why a model came out the way it did.

        Both routing arguments are positional-only: tools have parameters
        called ``name`` and ``model_name``, which would otherwise collide.
        """
        if agent_name not in self._agents:
            raise KeyError(f"no agent {agent_name!r}. Available: {sorted(self._agents)}")
        label = f"{agent_name}.{tool_name}"
        try:
            result = self._agents[agent_name].call_tool(tool_name, **kwargs)
        except Exception as exc:
            self.run.trace(label, kwargs, None, error=f"{type(exc).__name__}: {exc}")
            raise
        self.run.trace(label, kwargs, result)
        return result

    def describe(self) -> dict[str, Any]:
        """Inventory of agents and their tools -- useful for a smoke test."""
        return {
            name: {"description": a.description, "tools": sorted(a.tool_map())}
            for name, a in self._agents.items()
        }


def build_team(goal: str = "", label: str = "campaign", model: str | None = None) -> AgentTeam:
    """Assemble the phase-1 team inside a fresh run directory."""
    return AgentTeam(
        architect=build_architect_agent(model),
        trainer=build_trainer_agent(model),
        generator=build_generator_agent(model),
        run=Run.create(label, goal=goal),
    )
