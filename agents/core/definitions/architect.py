"""Architect agent: decide the energy-model architecture.

The decision is the LLM's. The tools exist only to give it ground truth --
what exists, how it performed, and whether a proposal is buildable. Note there
is no `propose_architecture` tool: the proposal is the agent's *output*,
produced as a structured `ArchSpec` via slapsci's `pydantic_model=`, not
something a function invents for it.
"""

from __future__ import annotations

from typing import Any

from .. import config as cfg
from ..schemas import ArchSpec, Layer
from ..tools import architecture, training
from .base import Agent, tool

__all__ = ["build_architect_agent", "ARCH_OUTPUT_MODEL"]

#: Hand this to slapsci as `pydantic_model=` so the agent's final answer is a
#: validated architecture rather than prose that has to be parsed.
ARCH_OUTPUT_MODEL = ArchSpec


@tool(
    """
    List every energy model the agents have trained so far, newest first,
    with its architecture summary and evaluation metrics.

    Call this before proposing anything: design relative to what already
    exists rather than from scratch.
    """
)
def list_trained_models() -> list[dict[str, Any]]:
    return architecture.list_trained_models()


@tool(
    """
    List the pre-existing reference weight files in data/energy_model_params/.

    These are the models the production ColabDesign pipeline currently uses.
    They are read-only and were trained outside the agent system.
    """
)
def list_reference_models() -> list[dict[str, Any]]:
    return architecture.list_reference_models()


@tool(
    """
    Full detail on one registered model: architecture, metrics, parameter
    count, and which run produced it.
    """
)
def inspect_model(model_name: str) -> dict[str, Any]:
    return architecture.inspect_model(model_name)


@tool(
    """
    Check whether a proposed architecture is buildable and sensibly sized.

    Returns the parameter count and a list of warnings. Pass
    n_training_examples so the parameter-to-data ratio can be checked --
    overfitting is the main risk in this data regime.

    Call this on any architecture you intend to propose, BEFORE proposing it.
    """
)
def validate_architecture(
    name: str,
    layer_units: list[int],
    layer_activations: list[str] | None = None,
    n_rounds: int = 2,
    seq_length: int = 15,
    likelihood: str = "negbinom",
    l2: float = 1e-4,
    n_training_examples: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    acts = layer_activations or (["tanh"] * (len(layer_units) - 1) + ["identity"])
    arch = ArchSpec(
        name=name,
        n_rounds=n_rounds,
        layers=[Layer(units=u, activation=a) for u, a in zip(layer_units, acts)],
        likelihood=likelihood,  # type: ignore[arg-type]
        seq_length=seq_length,
        regularization={"l2": l2},
        notes=notes,
    )
    return architecture.validate_architecture(arch, n_training_examples=n_training_examples)


@tool(
    """
    Compare registered models side by side on their metrics.

    Returns a table plus which model is best on each metric separately.
    Do not collapse this into a single winner without saying on what basis.
    """
)
def compare_models(model_names: list[str]) -> dict[str, Any]:
    return architecture.compare_models(model_names)


@tool(
    """
    List the measurement files available for training, with repo-relative
    paths. Use these exact paths when specifying a dataset -- do not guess
    filenames.
    """
)
def describe_available_data() -> dict[str, Any]:
    return training.describe_available_data()


SYSTEM_PROMPT = """\
You choose the architecture of the energy model that predicts binding fitness
for peptides at the influenza PA-PB1 interface.

Context: the model maps a one-hot 15-residue sequence to a scalar energy
(lower = better binder), trained on phage-display sequencing counts with a
negative-binomial likelihood. The existing production model is a 3-layer MLP
with two selection rounds, named in the style PNB_2R_3lay_negbinom.

The binding constraint is DATA, not compute. These datasets are small relative
to the input dimension (15 residues x 21 columns = 315; the alphabet carries a
trailing gap column). Overfitting is the failure mode that matters; a larger
network is almost never the answer.

Method:
1. Look at what exists first -- both the agent-trained models and the
   reference weights -- and at how much data is actually available.
2. Form a hypothesis about what to change and why. One change at a time, so
   the comparison is interpretable.
3. Validate the proposal before you commit to it, passing the real number of
   training examples.
4. Output the final architecture as a structured ArchSpec, with the `notes`
   field stating your reasoning and what you expect it to improve.

Be explicit about uncertainty. If the evidence does not distinguish between
two architectures, say so and propose the smaller one.
"""


def build_architect_agent(model: str | None = None) -> Agent:
    # llm.per_agent.architect in agents/config/default.yaml, falling back to llm.model.
    model = model or cfg.get("llm.per_agent.architect") or cfg.get("llm.model")
    return Agent(
        name="architect",
        description="Decides the energy-model architecture from what exists and what the data supports.",
        system_prompt=SYSTEM_PROMPT,
        tools=[
            list_trained_models,
            list_reference_models,
            inspect_model,
            validate_architecture,
            compare_models,
            describe_available_data,
        ],
        model=model,
    )
