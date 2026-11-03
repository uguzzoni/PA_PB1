"""Trainer agent: train and evaluate energy models.

This agent is the one that consumes real compute, so its tools are
deliberately pre-flight heavy: validate the dataset, estimate the cost, then
train. A failed training run that could have been caught by checking a file
path is pure waste.
"""

from __future__ import annotations

from typing import Any

from .. import config as cfg
from ..schemas import ArchSpec, DatasetSpec, Hyperparams, Layer
from ..tools import architecture, training
from .. import registry
from .base import Agent, tool

__all__ = ["build_trainer_agent"]


@tool(
    """
    List the measurement files available for training, with repo-relative
    paths. Use these exact paths -- never guess a filename.
    """
)
def describe_available_data() -> dict[str, Any]:
    return training.describe_available_data()


@tool(
    """
    Check a dataset specification before training: do the files exist, how
    many rows do they have. Always call this before train_model.
    """
)
def prepare_dataset(
    counts_file: str, affinity_file: str | None = None, val_fraction: float = 0.2
) -> dict[str, Any]:
    return training.prepare_dataset(
        DatasetSpec(
            counts_file=counts_file,
            affinity_file=affinity_file,
            val_fraction=val_fraction,
        )
    )


@tool(
    """
    Train an energy model and register it under `model_name`.

    This is the expensive call. Validate the dataset and the architecture
    first. Weights are written to the agent store only -- training never
    touches the production weights the ColabDesign pipeline uses.

    Choose a fresh, descriptive model_name; reusing a name is refused so that
    earlier models stay available for comparison.
    """
)
def train_model(
    model_name: str,
    layer_units: list[int],
    counts_file: str,
    layer_activations: list[str] | None = None,
    affinity_file: str | None = None,
    n_rounds: int = 2,
    seq_length: int = 15,
    likelihood: str = "negbinom",
    l2: float = 1e-4,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    notes: str = "",
) -> dict[str, Any]:
    acts = layer_activations or (["tanh"] * (len(layer_units) - 1) + ["identity"])
    arch = ArchSpec(
        name=model_name,
        n_rounds=n_rounds,
        layers=[Layer(units=u, activation=a) for u, a in zip(layer_units, acts)],
        likelihood=likelihood,  # type: ignore[arg-type]
        seq_length=seq_length,
        regularization={"l2": l2},
        notes=notes,
    )
    dataset = DatasetSpec(counts_file=counts_file, affinity_file=affinity_file)
    hyper = Hyperparams(
        epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed
    )
    return training.train_model(arch, dataset, model_name, hyper).model_dump()


@tool(
    """
    Evaluate a trained model against sequences with known measured values,
    returning Spearman and Pearson correlation.

    This is the cheapest and most informative check available: run it on the
    measured-affinity sequences before trusting any model.
    """
)
def evaluate_model(
    model_name: str, sequences: list[str], measured: list[float]
) -> dict[str, Any]:
    return training.evaluate_model(model_name, sequences, measured).model_dump(exclude_none=True)


@tool(
    """
    Parameter count and sizing warnings for an architecture, without training
    it. Use to sanity-check cost before committing.
    """
)
def estimate_cost(
    layer_units: list[int], seq_length: int = 15, n_training_examples: int | None = None
) -> dict[str, Any]:
    acts = ["tanh"] * (len(layer_units) - 1) + ["identity"]
    arch = ArchSpec(
        name="estimate",
        layers=[Layer(units=u, activation=a) for u, a in zip(layer_units, acts)],
        seq_length=seq_length,
    )
    return architecture.validate_architecture(arch, n_training_examples=n_training_examples)


@tool(
    """
    List models already trained, so you do not retrain something that exists.
    """
)
def list_trained_models() -> list[dict[str, Any]]:
    return architecture.list_trained_models()


@tool(
    """
    Explain how to promote a trained model into production use by the
    ColabDesign pipeline. This returns INSTRUCTIONS for a human to run --
    promotion is deliberately not automated and you cannot perform it.
    """
)
def how_to_promote(model_name: str) -> str:
    return registry.promotion_instructions(model_name)


SYSTEM_PROMPT = """\
You train energy models that predict binding fitness for peptides at the
influenza PA-PB1 interface, given an architecture decided upstream.

Training costs real time. Work in this order, always:
1. Check what has already been trained -- do not repeat an existing run.
2. List the available data and verify the exact file paths.
3. Check the dataset loads and see how many rows it has.
4. Estimate the parameter count against that row count. If the ratio looks
   like overfitting, say so before training rather than after.
5. Train, then evaluate against measured affinities.

Report the metrics plainly, including when they are poor. A model with a
Spearman correlation near zero is an important finding, not something to
present as a partial success. If you believe the architecture is the problem
rather than the training, say which part and why.

You cannot promote a model into production; that is a human decision. If a
model looks good enough to use, say so and explain how to promote it.
"""


def build_trainer_agent(model: str | None = None) -> Agent:
    # llm.per_agent.trainer in agents/config/default.yaml, falling back to llm.model.
    model = model or cfg.get("llm.per_agent.trainer") or cfg.get("llm.model")
    return Agent(
        name="trainer",
        description="Trains and evaluates energy models from a given architecture.",
        system_prompt=SYSTEM_PROMPT,
        tools=[
            describe_available_data,
            prepare_dataset,
            estimate_cost,
            list_trained_models,
            train_model,
            evaluate_model,
            how_to_promote,
        ],
        model=model,
    )
