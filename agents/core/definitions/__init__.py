"""Agent definitions: system prompts and permitted tool sets."""

from .architect import build_architect_agent
from .base import Agent, Tool, bind_to_slapsci, tool
from .generator import build_generator_agent
from .orchestrator import AgentTeam, build_team
from .trainer import build_trainer_agent

__all__ = [
    "Agent",
    "Tool",
    "tool",
    "bind_to_slapsci",
    "build_architect_agent",
    "build_trainer_agent",
    "build_generator_agent",
    "build_team",
    "AgentTeam",
]
