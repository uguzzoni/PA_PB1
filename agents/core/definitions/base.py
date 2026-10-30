"""Framework-agnostic agent and tool definitions.

The tool layer below ``agents/core/tools/`` is plain Python functions.  This
module turns them into LLM-callable tools *without* importing slapsci, so the
whole system stays unit-testable with no provider, no API key and no network.

:func:`bind_to_slapsci` is the only place that knows slapsci exists. If you
ever swap framework, that function is the entire blast radius.

.. note::
   **No generic filesystem or shell tool is defined anywhere in this package,
   and none should be.** The compartmentalisation guarantee in
   ``workspace.py`` is only as strong as this rule: an agent cannot reach what
   no tool exposes. If a new capability seems to need raw paths, add a
   domain tool that takes an identifier instead.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from pydantic import BaseModel, create_model

__all__ = ["Tool", "Agent", "tool", "build_params_model", "bind_to_slapsci"]

#: Parameters that are wiring, not model-facing arguments.
_INTERNAL = {"runner", "run", "prefer_julia", "overwrite", "use_julia"}


def build_params_model(fn: Callable, exclude: Iterable[str] = ()) -> type[BaseModel]:
    """Derive a pydantic model for a function's call signature.

    Keeps tool modules free of boilerplate: the functions stay ordinary Python
    and the JSON schema shown to the LLM is generated from their annotations.
    """
    skip = _INTERNAL | set(exclude)
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # unresolvable forward refs -- fall back to raw annotations
        hints = {k: v.annotation for k, v in sig.parameters.items()}

    fields: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name in skip or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = hints.get(name, Any)
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[name] = (ann, default)

    return create_model(f"{fn.__name__.title().replace('_', '')}Params", **fields)


@dataclass
class Tool:
    """One LLM-callable capability.

    ``description`` is read by the model, so it must say *when* to use the
    tool, not just what it does.
    """

    name: str
    description: str
    fn: Callable
    params_model: type[BaseModel]

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.params_model.model_json_schema()

    def call(self, **kwargs: Any) -> Any:
        """Validate arguments against the schema, then invoke.

        Validation here is not ceremony: it is the layer that turns a model's
        malformed guess into a clear error the agent can recover from, rather
        than a TypeError deep inside numpy.
        """
        validated = self.params_model.model_validate(kwargs)
        return self.fn(**validated.model_dump(exclude_unset=False))

    def to_openai(self) -> dict[str, Any]:
        """OpenAI / slapsci-compatible function-tool description."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def tool(description: str, *, name: str | None = None, exclude: Iterable[str] = ()) -> Callable:
    """Decorator turning a plain function into a :class:`Tool`.

    >>> @tool("Score sequences with a trained energy model. Lower is better.")
    ... def predict_fitness(sequences: list[str], model_name: str) -> list[dict]:
    ...     ...
    """

    def wrap(fn: Callable) -> Tool:
        return Tool(
            name=name or fn.__name__,
            description=inspect.cleandoc(description),
            fn=fn,
            params_model=build_params_model(fn, exclude),
        )

    return wrap


@dataclass
class Agent:
    """An LLM agent: a system prompt plus the tools it is allowed to call.

    An agent's tool list *is* its permission set. Keep it minimal -- an agent
    given every tool will use the wrong one.
    """

    name: str
    description: str
    system_prompt: str
    tools: list[Tool] = field(default_factory=list)
    model: str | None = None

    def tool_map(self) -> dict[str, Tool]:
        return {t.name: t for t in self.tools}

    def call_tool(self, tool_name: str, /, **kwargs: Any) -> Any:
        """Invoke one of this agent's tools.

        ``tool_name`` is positional-only on purpose: several tools take a
        ``name`` argument of their own, and a keyword parameter here would
        collide with it.
        """
        tools = self.tool_map()
        if tool_name not in tools:
            raise KeyError(
                f"agent {self.name!r} has no tool {tool_name!r}. Available: {sorted(tools)}"
            )
        return tools[tool_name].call(**kwargs)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai() for t in self.tools]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic() for t in self.tools]


# --------------------------------------------------------------------------
# slapsci binding -- the only framework-aware code in the package
# --------------------------------------------------------------------------
def bind_to_slapsci(agent: Agent, provider_name: str = "anthropic", model: str | None = None):
    """Instantiate a slapsci provider wired to this agent's tools.

    .. note::
       Written against the public API in the slapsci README
       (``get_llm_provider(provider, model)``). The exact keyword for
       registering tools is **not** documented there -- check
       ``examples/tools_usage/example_tools_usage.py`` in the slapsci repo and
       adjust the ``tools=`` keyword below. Everything else in this package is
       independent of that detail.
    """
    try:
        from slapsci import get_llm_provider  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "slapsci is not installed. The rest of this package does not need it: "
            "tools can be called directly via Agent.call_tool() for testing."
        ) from exc

    return get_llm_provider(
        provider_name,
        model or agent.model or "claude-sonnet-4-6",
        tools=agent.to_openai_tools(),   # <- verify keyword against the slapsci examples
        system_prompt=agent.system_prompt,
    )
