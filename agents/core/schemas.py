"""I/O contracts for the agent system.

Every tool takes a pydantic model in and returns a pydantic model out.  Three
things depend on this:

1. **Tool schemas for the LLM.**  ``model_json_schema()`` is what slapsci (or
   any tool-calling provider) shows the model.  Field descriptions here are
   read by the LLM -- write them as instructions, not as notes to yourself.
2. **The Julia boundary.**  These serialise to the ``config.json`` /
   ``result.json`` exchanged with ``agents/julia/*.jl``.
3. **Structured output.**  ``ArchSpec`` is handed to slapsci's
   ``pydantic_model=`` so the Architect agent cannot emit an unusable design.

.. important::
   **Sign convention.**  The energy model is a *cost*: lower is better.  This
   matches ``loss_total = loss_af + energy_weight * energy`` in
   ``src/generative_protocols/model_energy_guidance.py``, which is minimised.
   Every sampler here therefore **minimises** energy by default.  If the Julia
   side returns a fitness where *higher* is better, set
   ``objective="maximize"`` -- do not flip the sign silently somewhere in the
   middle of the stack.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "AA_ALPHABET",
    "Activation",
    "Layer",
    "ArchSpec",
    "Hyperparams",
    "DatasetSpec",
    "TrainConfig",
    "EvalMetrics",
    "TrainResult",
    "ModelEntry",
    "SeqConstraints",
    "SampleConfig",
    "SeqRecord",
    "SampleResult",
    "FitnessScore",
    "DiversityMetrics",
]


#: Canonical amino-acid order used for one-hot encoding.
#:
#: .. warning::
#:    This **must** match the order used when the Julia model was trained.
#:    If it does not, ``predict_fitness`` returns confident nonsense rather
#:    than failing.  Verify against ``PhageNegBinom.jl`` before trusting any
#:    number out of the Python fast path -- ``tests/test_fitness.py`` has a
#:    round-trip check for this.
AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"

Activation = Literal["tanh", "relu", "identity", "sigmoid", "softplus"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
class Layer(_Base):
    """One dense layer of the energy MLP."""

    units: int = Field(..., ge=1, le=4096, description="Number of output units.")
    activation: Activation = Field(
        "tanh", description="Nonlinearity applied after the affine map."
    )


class ArchSpec(_Base):
    """A fully specified energy-model architecture.

    This is the contract between the Architect agent (which *decides* it) and
    the Trainer agent (which *builds* it).  The naming mirrors the existing
    convention, e.g. ``PNB_2R_3lay_negbinom`` -> two rounds, three layers,
    negative-binomial likelihood.
    """

    name: str = Field(
        ...,
        description=(
            "Short identifier for this architecture. Letters, digits, dot, "
            "dash and underscore only; it becomes a directory name."
        ),
    )
    n_rounds: int = Field(
        2, ge=1, le=5, description="Number of selection rounds modelled (the 'R' in PNB_2R)."
    )
    layers: list[Layer] = Field(
        ...,
        min_length=1,
        description=(
            "Dense layers in order. The final layer must have units=1 and "
            "activation='identity': it emits the scalar energy."
        ),
    )
    likelihood: Literal["negbinom", "poisson", "gaussian"] = Field(
        "negbinom", description="Observation model for the sequencing counts."
    )
    input_encoding: Literal["onehot", "soft_logits"] = Field(
        "onehot",
        description=(
            "'onehot' for discrete sequences; 'soft_logits' for the "
            "differentiable path used by ColabDesign energy guidance."
        ),
    )
    seq_length: int = Field(
        15, ge=1, le=200, description="Binder length in residues. 15 for the PA-PB1 campaign."
    )
    regularization: dict[str, float] = Field(
        default_factory=lambda: {"l2": 1e-4},
        description="Regularisation terms, e.g. {'l2': 1e-4}.",
    )
    notes: str = Field(
        "", description="Free-text rationale. The Architect agent should say *why* it chose this."
    )

    @field_validator("layers")
    @classmethod
    def _last_layer_is_scalar(cls, v: list[Layer]) -> list[Layer]:
        if v[-1].units != 1:
            raise ValueError(
                f"final layer must have units=1 (got {v[-1].units}): the model emits a scalar energy"
            )
        if v[-1].activation != "identity":
            raise ValueError(
                f"final layer must have activation='identity' (got {v[-1].activation!r}): "
                "a squashed output cannot represent an unbounded energy"
            )
        return v

    @property
    def input_dim(self) -> int:
        return self.seq_length * len(AA_ALPHABET)

    def n_params(self) -> int:
        """Total trainable parameters, including biases."""
        total = 0
        prev = self.input_dim
        for layer in self.layers:
            total += prev * layer.units + layer.units
            prev = layer.units
        return total


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
class Hyperparams(_Base):
    epochs: int = Field(200, ge=1, le=100_000)
    learning_rate: float = Field(1e-3, gt=0, le=1.0)
    batch_size: int = Field(256, ge=1)
    early_stopping_patience: int = Field(
        20, ge=0, description="Epochs without validation improvement before stopping. 0 disables."
    )
    seed: int = Field(0, description="RNG seed; set explicitly so runs are reproducible.")


class DatasetSpec(_Base):
    """Which measurements to train on.

    Paths are **repository-relative** and resolved through
    ``workspace.read_path``; absolute paths are rejected.
    """

    counts_file: str = Field(
        ..., description="Repo-relative path to the sequencing counts used for the count likelihood."
    )
    affinity_file: str | None = Field(
        None, description="Repo-relative path to measured affinities, for evaluation."
    )
    val_fraction: float = Field(0.2, ge=0.0, lt=1.0, description="Held-out fraction.")
    split_seed: int = Field(0, description="Seed for the train/validation split.")

    @field_validator("counts_file", "affinity_file")
    @classmethod
    def _must_be_relative(cls, v: str | None) -> str | None:
        if v is not None and (v.startswith("/") or ".." in v):
            raise ValueError(f"path must be repository-relative and contain no '..': {v!r}")
        return v


class TrainConfig(_Base):
    """Everything ``agents/julia/train.jl`` needs."""

    model_name: str = Field(..., description="Name under which to register the trained model.")
    arch: ArchSpec
    dataset: DatasetSpec
    hyperparams: Hyperparams = Field(default_factory=Hyperparams)


class EvalMetrics(_Base):
    spearman: float | None = Field(None, description="Rank correlation predicted vs measured.")
    pearson: float | None = None
    val_loss: float | None = None
    n: int = Field(0, description="Number of sequences evaluated.")


class TrainResult(_Base):
    model_name: str
    weights_path: str = Field(..., description="Path to the exported JSON weights, store-relative.")
    arch: ArchSpec
    metrics: EvalMetrics
    n_params: int
    duration_s: float = 0.0


class ModelEntry(_Base):
    """One row of ``agents/store/models/registry.json``."""

    name: str
    arch: ArchSpec
    metrics: EvalMetrics
    weights_path: str
    n_params: int
    run_id: str | None = None
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted: bool = Field(
        False,
        description=(
            "True once a human has copied these weights into "
            "data/energy_model_params/ for use by the ColabDesign pipeline. "
            "Agents never set this."
        ),
    )


# --------------------------------------------------------------------------
# Sequence generation
# --------------------------------------------------------------------------
class SeqConstraints(_Base):
    """Hard constraints applied to every generated sequence."""

    forbidden_aa: str = Field(
        "C",
        description=(
            "Amino acids that may never appear, as one-letter codes. "
            "Cysteine is excluded by default to avoid spurious disulfides."
        ),
    )
    fixed_positions: dict[int, str] = Field(
        default_factory=dict,
        description="0-based position -> required amino acid. Use to pin known anchor residues.",
    )
    max_identity_to_seeds: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Reject sequences more identical than this to any seed sequence. "
            "Set it to force novelty away from known binders."
        ),
    )


class SampleConfig(_Base):
    """Everything a sampler needs to produce sequences from an energy model."""

    model_name: str = Field(..., description="Registered model to sample from.")
    n: int = Field(100, ge=1, le=100_000, description="Number of sequences to return.")
    length: int = Field(15, ge=1, le=200)
    method: Literal["mcmc", "gradient", "greedy"] = Field(
        "mcmc",
        description=(
            "'mcmc': Metropolis over point mutations (robust default). "
            "'gradient': ascent on soft logits then discretise (needs a differentiable model). "
            "'greedy': directed evolution from the best seeds."
        ),
    )
    objective: Literal["minimize", "maximize"] = Field(
        "minimize",
        description=(
            "Whether good sequences have LOW or HIGH model output. The energy "
            "model is a cost, so 'minimize' is correct unless you know otherwise."
        ),
    )
    temperature: float = Field(
        1.0, gt=0, description="Metropolis temperature. Lower = greedier, less diverse."
    )
    n_steps: int = Field(1000, ge=1, description="MCMC steps per chain.")
    seed: int = Field(0)
    seeds: list[str] = Field(
        default_factory=list,
        description="Optional starting sequences. Empty means random initialisation.",
    )
    constraints: SeqConstraints = Field(default_factory=SeqConstraints)


class SeqRecord(_Base):
    seq: str
    energy: float = Field(..., description="Raw model output. Lower is better by convention.")
    source: str = Field("", description="Which sampler/model produced it.")

    @field_validator("seq")
    @classmethod
    def _valid_aa(cls, v: str) -> str:
        bad = set(v) - set(AA_ALPHABET)
        if bad:
            raise ValueError(f"sequence contains non-standard residues: {sorted(bad)}")
        return v


class SampleResult(_Base):
    config: SampleConfig
    sequences: list[SeqRecord]
    accepted_fraction: float | None = Field(
        None, description="MCMC acceptance rate. Very low or very high means the temperature is off."
    )


class FitnessScore(_Base):
    seq: str
    energy: float


class DiversityMetrics(_Base):
    """Diversity is a first-class criterion, not an afterthought.

    A sampler optimising energy alone collapses onto one motif, and an LLM
    reading only mean energy will happily report success.  The Generator agent
    is instructed to check these alongside fitness.
    """

    n: int
    mean_hamming: float = Field(..., description="Mean pairwise Hamming distance.")
    min_hamming: int = Field(..., description="Closest pair. 0 means exact duplicates.")
    n_unique: int
    n_clusters: int = Field(..., description="Clusters at the identity threshold used.")
    mean_position_entropy: float = Field(
        ..., description="Mean per-position Shannon entropy in bits. Max is log2(20) = 4.32."
    )
