"""Generator agent: produce candidate binders from a trained energy model.

Phase 1 target. Runs entirely on CPU with the Python fast path, so this agent
is testable end-to-end before any Julia or GPU work exists.

The adapter functions below present *flat* signatures to the LLM and assemble
the nested config objects internally. Models call flat tools far more reliably
than deeply nested ones.
"""

from __future__ import annotations

from typing import Any

from .. import config as cfg
from ..schemas import SampleConfig, SeqConstraints, SeqRecord
from ..tools import analysis, fitness
from .base import Agent, tool

__all__ = ["build_generator_agent"]

# Read once, at import, so the concrete numbers appear in the tool schemas the
# LLM is shown. See agents/config/default.yaml.
_N = int(cfg.get("sampling.n", 50))
_TEMPERATURE = float(cfg.get("sampling.temperature", 1.0))
_N_STEPS = int(cfg.get("sampling.n_steps", 1000))
_MIN_MEAN_HAMMING = float(cfg.get("sampling.min_mean_hamming", 3.0))
_SEQ_LENGTH = int(cfg.get("defaults.seq_length", 15))
_FORBIDDEN_AA = str(cfg.get("defaults.forbidden_aa", "C"))


@tool(
    """
    Score peptide sequences with a trained energy model.

    Returns one energy per sequence. LOWER energy means a BETTER predicted
    binder -- the model is a cost, not a score. Use this to check candidates
    before and after generation, and to compare against known binders.
    """
)
def predict_fitness(sequences: list[str], model_name: str) -> list[dict[str, Any]]:
    return [s.model_dump() for s in fitness.predict_fitness(sequences, model_name)]


@tool(
    """
    Generate new candidate sequences by sampling from a trained energy model.

    Runs one Metropolis chain per requested sequence and returns the best state
    each chain visited. Raise `temperature` or `n_steps` for more exploration;
    lower `temperature` for greedier, less diverse output.

    ALWAYS follow this with check_diversity: a sampler optimising energy alone
    frequently collapses onto a single motif, and mean energy will look
    excellent while every sequence is nearly identical.
    """
)
def generate_sequences(
    model_name: str,
    n: int = _N,
    length: int = _SEQ_LENGTH,
    temperature: float = _TEMPERATURE,
    n_steps: int = _N_STEPS,
    seed: int = 0,
    seed_sequences: list[str] | None = None,
    forbidden_aa: str = _FORBIDDEN_AA,
) -> dict[str, Any]:
    config = SampleConfig(
        model_name=model_name,
        n=n,
        length=length,
        method="mcmc",
        temperature=temperature,
        n_steps=n_steps,
        seed=seed,
        seeds=seed_sequences or [],
        constraints=SeqConstraints(forbidden_aa=forbidden_aa),
    )
    result = fitness.sample_sequences(config)
    return {
        "sequences": [r.model_dump() for r in result.sequences],
        "accepted_fraction": result.accepted_fraction,
        "hint": (
            "acceptance below ~0.05 means the temperature is too low (chains are stuck); "
            "above ~0.8 means it is too high (the chain is a random walk)."
        ),
    }


@tool(
    """
    Measure how varied a set of sequences is.

    Report these alongside fitness, never instead of it. Warning signs:
    n_unique much smaller than n, min_hamming of 0 (exact duplicates),
    n_clusters near 1 (all one motif), or mean_position_entropy near 0.
    Maximum possible entropy is 4.32 bits.
    """
)
def check_diversity(sequences: list[str], identity_threshold: float = 0.8) -> dict[str, Any]:
    metrics = analysis.diversity_metrics(sequences, identity_threshold=identity_threshold)
    out = metrics.model_dump()
    # Threshold from sampling.min_mean_hamming in agents/config/default.yaml.
    out["collapsed"] = metrics.n_clusters <= 1 or metrics.mean_hamming < _MIN_MEAN_HAMMING
    out["min_mean_hamming"] = _MIN_MEAN_HAMMING
    return out


@tool(
    """
    Rank sequences by predicted energy and return the best ones.

    Lower energy first. Use top_k to shortlist candidates.
    """
)
def rank_candidates(
    sequences: list[str], model_name: str, top_k: int = 20
) -> list[dict[str, Any]]:
    scored = fitness.predict_fitness(sequences, model_name)
    records = [SeqRecord(seq=s.seq, energy=s.energy, source=model_name) for s in scored]
    return [r.model_dump() for r in analysis.rank_sequences(records, top_k=top_k)]


@tool(
    """
    Save a final set of candidate sequences under a name, as JSON and CSV.

    Use this only for results worth keeping. Files land in the agent store;
    you cannot write anywhere else.
    """
)
def save_candidates(
    sequences: list[str], energies: list[float], name: str, source: str = ""
) -> dict[str, str]:
    if len(sequences) != len(energies):
        raise ValueError(f"{len(sequences)} sequences but {len(energies)} energies")
    records = [
        SeqRecord(seq=s, energy=e, source=source) for s, e in zip(sequences, energies)
    ]
    return analysis.save_sequences(records, name)


@tool(
    """
    Read known sequences from a file in the repository, e.g. previously measured
    binders. Use these as seeds for generation or as a baseline to compare
    generated candidates against.

    Example paths: 'data/input_sequences.txt',
    'data/seqs_aff_alberto_darren.txt'.
    """
)
def load_reference_sequences(repo_relative_path: str) -> list[str]:
    return analysis.read_seed_sequences(repo_relative_path)


SYSTEM_PROMPT = """\
You design candidate peptide binders by sampling from a trained energy model.

The target is the influenza PA-PB1 polymerase interface; binders are 15
residues unless told otherwise. This is in-silico candidate generation: your
output is a shortlist for downstream structural scoring, not a conclusion.

Sign convention: the model emits an ENERGY. Lower is better. Never report a
high energy as a good result.

Method:
1. Establish a baseline. Load known binders and score them, so you know what a
   good energy actually looks like for this model. Do not judge generated
   sequences against an absolute number you assumed.
2. Generate, then immediately check diversity. Fitness and diversity are both
   first-class. A set with excellent mean energy and one motif is a failure,
   and you must say so rather than reporting the energy alone.
3. If the chains collapsed, raise the temperature or the step count and
   regenerate. If acceptance is very low, the temperature is too low.
4. Shortlist with rank_candidates, then save.

Report honestly: give mean and best energy, the diversity metrics, the
acceptance fraction, and how the candidates compare to the known binders. If
the run did not produce anything better than the known binders, say that
plainly -- it is a useful result, not a failure to hide.
"""


def build_generator_agent(model: str | None = None) -> Agent:
    # llm.per_agent.generator in agents/config/default.yaml, falling back to llm.model.
    model = model or cfg.get("llm.per_agent.generator") or cfg.get("llm.model")
    return Agent(
        name="generator",
        description="Samples candidate binders from a trained fitness model and assesses them.",
        system_prompt=SYSTEM_PROMPT,
        tools=[
            predict_fitness,
            generate_sequences,
            check_diversity,
            rank_candidates,
            save_candidates,
            load_reference_sequences,
        ],
        model=model,
    )
