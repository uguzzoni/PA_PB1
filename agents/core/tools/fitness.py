"""Fitness tools: score sequences, and sample new ones from the energy model.

Two execution paths, deliberately:

* **Python fast path** (this module) -- loads the *exported JSON weights* and
  runs the 3-layer MLP in numpy.  No Julia, no GPU, milliseconds per batch.
  This is what makes phase 1 runnable today, and it is the same forward pass
  ``src/generative_protocols/model_energy_guidance.py`` injects into the
  ColabDesign loss, so agreement between the two is a real correctness check.
* **Julia path** -- ``agents/julia/sample.jl`` backed by ``PhageMC.jl``, for
  samplers that already exist there and should not be reimplemented.

The MCMC below is a reference implementation, not a replacement for
``PhageMC.jl``.  It exists so the Generator agent can be tested end-to-end
before any Julia is wired up; swap it out via ``SampleConfig.method`` routing
in :func:`sample_sequences` once the real entry point works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .. import registry, workspace as ws
from ..execution import Runner, default_runner
from ..schemas import (
    AA_ALPHABET,
    FitnessScore,
    SampleConfig,
    SampleResult,
    SeqConstraints,
    SeqRecord,
)

__all__ = [
    "EnergyModel",
    "inspect_weights_file",
    "load_energy_model",
    "encode_onehot",
    "predict_fitness",
    "sample_sequences",
]

_AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
_N_AA = len(AA_ALPHABET)


# ==========================================================================
# Weight loading
# ==========================================================================
def inspect_weights_file(path: str | Path) -> dict[str, Any]:
    """Describe the structure of an exported weights file without interpreting it.

    Run this first against a real
    ``data/energy_model_params/*_energy_model_weights.json``.  The loader below
    guesses among a few plausible layouts; this tells you which one you have so
    the guess can be replaced with certainty.

    >>> inspect_weights_file("data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json")
    """
    p = Path(path)
    if not p.is_absolute():
        p = ws.read_path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    def describe(obj: Any, depth: int = 0) -> Any:
        if depth > 3:
            return "..."
        if isinstance(obj, dict):
            return {k: describe(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            arr = np.asarray(obj, dtype=object)
            try:
                num = np.asarray(obj, dtype=float)
                return f"array shape={num.shape} dtype=float"
            except (ValueError, TypeError):
                return f"list len={len(obj)} (ragged) first={describe(obj[0], depth + 1) if obj else None}"
        return type(obj).__name__

    return {"path": str(p), "top_level_keys": list(raw) if isinstance(raw, dict) else None,
            "structure": describe(raw)}


def _to_float_array(obj: Any) -> np.ndarray:
    return np.asarray(obj, dtype=np.float64)


def _extract_layers(raw: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pull (W, b) pairs out of whatever layout the export used.

    Handles the three shapes we are likely to meet.  If none match, the error
    lists what was actually found so the fix is one edit, not an investigation.
    """
    # Layout A: {"layers": [{"W": ..., "b": ...}, ...]}
    if isinstance(raw, dict) and isinstance(raw.get("layers"), list):
        out = []
        for i, layer in enumerate(raw["layers"]):
            w_key = next((k for k in ("W", "weight", "weights", "w") if k in layer), None)
            b_key = next((k for k in ("b", "bias", "biases") if k in layer), None)
            if w_key is None:
                raise ValueError(f"layer {i} has no weight key; keys were {list(layer)}")
            out.append((
                _to_float_array(layer[w_key]),
                _to_float_array(layer[b_key]) if b_key else np.zeros(1),
            ))
        return out

    # Layout B: {"W1": ..., "b1": ..., "W2": ...}
    if isinstance(raw, dict) and any(k.lower().startswith("w") and k[1:].isdigit() for k in raw):
        idx = sorted(
            {int(k[1:]) for k in raw if k.lower().startswith("w") and k[1:].isdigit()}
        )
        out = []
        for i in idx:
            w = next(raw[k] for k in raw if k.lower() == f"w{i}")
            b = next((raw[k] for k in raw if k.lower() == f"b{i}"), None)
            out.append((_to_float_array(w), _to_float_array(b) if b is not None else np.zeros(1)))
        return out

    # Layout C: {"weights": [...], "biases": [...]}
    if isinstance(raw, dict) and "weights" in raw and "biases" in raw:
        return [
            (_to_float_array(w), _to_float_array(b))
            for w, b in zip(raw["weights"], raw["biases"])
        ]

    raise ValueError(
        "unrecognised weights layout. Run inspect_weights_file() on this file and "
        "extend _extract_layers() for the layout you actually have. "
        f"Top-level keys: {list(raw) if isinstance(raw, dict) else type(raw).__name__}"
    )


_ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "identity": lambda x: x,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "softplus": lambda x: np.logaddexp(0.0, x),
}


@dataclass
class EnergyModel:
    """A loaded MLP energy model. Lower output = better binder, by convention."""

    layers: list[tuple[np.ndarray, np.ndarray]]
    activations: list[str]
    seq_length: int
    name: str = ""

    @property
    def input_dim(self) -> int:
        return self.seq_length * _N_AA

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. ``x`` is (batch, input_dim); returns (batch,)."""
        h = np.atleast_2d(x)
        for (w, b), act in zip(self.layers, self.activations):
            h = _ACTIVATIONS[act](h @ w + b)
        if h.shape[-1] != 1:
            raise ValueError(f"model output is not scalar: final shape {h.shape}")
        return h.reshape(-1)


def load_energy_model(
    weights_path: str | Path,
    *,
    seq_length: int = 15,
    activations: Sequence[str] | None = None,
    name: str = "",
) -> EnergyModel:
    """Load exported JSON weights into a numpy forward pass.

    Julia is column-major and ``Dense(in, out)`` stores its weight as
    ``(out, in)``, so an exported matrix may arrive in either orientation.
    Rather than guess, this transposes whichever way makes the dimensions chain
    from ``seq_length * 20`` and raises if neither does.
    """
    p = Path(weights_path)
    if not p.is_absolute():
        p = ws.read_path(weights_path) if (ws.REPO_ROOT / weights_path).exists() \
            else ws.write_path(weights_path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    pairs = _extract_layers(raw)

    fixed: list[tuple[np.ndarray, np.ndarray]] = []
    dim = seq_length * _N_AA
    for i, (w, b) in enumerate(pairs):
        if w.ndim != 2:
            raise ValueError(f"layer {i}: expected a 2-D weight, got shape {w.shape}")
        if w.shape[0] == dim:
            w_fixed = w
        elif w.shape[1] == dim:
            w_fixed = w.T
        else:
            raise ValueError(
                f"layer {i}: weight shape {w.shape} chains with neither orientation from "
                f"input dim {dim}. Check seq_length (got {seq_length}) and the alphabet "
                f"ordering in schemas.AA_ALPHABET."
            )
        out_dim = w_fixed.shape[1]
        b_fixed = np.asarray(b, dtype=np.float64).reshape(-1)
        if b_fixed.size not in (out_dim, 1):
            raise ValueError(f"layer {i}: bias size {b_fixed.size} != output dim {out_dim}")
        fixed.append((w_fixed, b_fixed))
        dim = out_dim

    if activations is None:
        # Convention from CLAUDE.md: 3-layer MLP, nonlinear hidden, linear output.
        activations = ["tanh"] * (len(fixed) - 1) + ["identity"]
    if len(activations) != len(fixed):
        raise ValueError(f"{len(activations)} activations for {len(fixed)} layers")

    return EnergyModel(fixed, list(activations), seq_length, name or p.stem)


# ==========================================================================
# Encoding
# ==========================================================================
def encode_onehot(sequences: Iterable[str], seq_length: int | None = None) -> np.ndarray:
    """One-hot encode sequences to (batch, L*20), row-major over positions.

    .. warning::
       The residue order is :data:`agents.core.schemas.AA_ALPHABET`. It must
       match the order used at training time in Julia.
    """
    seqs = list(sequences)
    if not seqs:
        return np.zeros((0, 0))
    L = seq_length or len(seqs[0])
    out = np.zeros((len(seqs), L, _N_AA), dtype=np.float64)
    for i, s in enumerate(seqs):
        if len(s) != L:
            raise ValueError(f"sequence {i} has length {len(s)}, expected {L}")
        for j, aa in enumerate(s):
            try:
                out[i, j, _AA_INDEX[aa]] = 1.0
            except KeyError:
                raise ValueError(f"sequence {i} position {j}: unknown residue {aa!r}") from None
    return out.reshape(len(seqs), L * _N_AA)


# ==========================================================================
# Tools
# ==========================================================================
def predict_fitness(
    sequences: list[str],
    model_name: str | None = None,
    *,
    weights_path: str | None = None,
    seq_length: int | None = None,
) -> list[FitnessScore]:
    """Score sequences with a trained energy model. CPU only, no Julia.

    Give either a registered ``model_name`` or an explicit ``weights_path``
    (repo-relative, e.g. a file in ``data/energy_model_params/``).
    """
    if not sequences:
        return []
    if model_name is None and weights_path is None:
        raise ValueError("provide either model_name or weights_path")

    if model_name is not None:
        entry = registry.get_model(model_name)
        path = entry.weights_path
        L = seq_length or entry.arch.seq_length
    else:
        path = weights_path  # type: ignore[assignment]
        L = seq_length or len(sequences[0])

    model = load_energy_model(path, seq_length=L, name=model_name or "")
    energies = model.energy(encode_onehot(sequences, L))
    return [FitnessScore(seq=s, energy=float(e)) for s, e in zip(sequences, energies)]


# -- MCMC reference implementation -----------------------------------------
def _allowed_alphabet(constraints: SeqConstraints) -> np.ndarray:
    forbidden = set(constraints.forbidden_aa.upper())
    idx = [i for aa, i in _AA_INDEX.items() if aa not in forbidden]
    if not idx:
        raise ValueError("constraints forbid every amino acid")
    return np.array(sorted(idx), dtype=np.int64)


def _apply_fixed(mat: np.ndarray, constraints: SeqConstraints) -> np.ndarray:
    for pos, aa in constraints.fixed_positions.items():
        if not 0 <= pos < mat.shape[1]:
            raise ValueError(f"fixed position {pos} out of range for length {mat.shape[1]}")
        mat[:, pos] = _AA_INDEX[aa.upper()]
    return mat


def _decode(mat: np.ndarray) -> list[str]:
    return ["".join(AA_ALPHABET[i] for i in row) for row in mat]


def sample_mcmc_python(
    model: EnergyModel,
    config: SampleConfig,
) -> SampleResult:
    """Metropolis sampling over point mutations, one chain per requested sequence.

    Chains are stepped in lockstep so every proposal batch is a single
    vectorised forward pass. Returns the best state visited by each chain,
    which is both greedier and more diverse than returning final states.
    """
    rng = np.random.default_rng(config.seed)
    L, n = config.length, config.n
    alphabet = _allowed_alphabet(config.constraints)
    sign = 1.0 if config.objective == "minimize" else -1.0
    mutable = [p for p in range(L) if p not in config.constraints.fixed_positions]
    if not mutable:
        raise ValueError("every position is fixed; nothing to sample")

    # Initialise: seeds where given, random otherwise.
    state = rng.choice(alphabet, size=(n, L))
    for i, s in enumerate(config.seeds[:n]):
        if len(s) != L:
            raise ValueError(f"seed {i!r} has length {len(s)}, expected {L}")
        state[i] = [_AA_INDEX[a] for a in s]
    state = _apply_fixed(state, config.constraints)

    def energies_of(mat: np.ndarray) -> np.ndarray:
        return sign * model.energy(encode_onehot(_decode(mat), L))

    cur_e = energies_of(state)
    best_state, best_e = state.copy(), cur_e.copy()
    accepted = 0

    for _ in range(config.n_steps):
        pos = rng.choice(mutable, size=n)
        new_aa = alphabet[rng.integers(0, len(alphabet), size=n)]
        proposal = state.copy()
        proposal[np.arange(n), pos] = new_aa

        prop_e = energies_of(proposal)
        delta = prop_e - cur_e
        # numpy evaluates both sides of `|`, so the exponent is computed even
        # where delta <= 0 (accepted outright). Clip at 0 to keep it finite:
        # a very low temperature would otherwise overflow to inf.
        accept_prob = np.exp(np.clip(-delta / config.temperature, -700.0, 0.0))
        accept = (delta <= 0) | (rng.random(n) < accept_prob)

        state[accept] = proposal[accept]
        cur_e[accept] = prop_e[accept]
        accepted += int(accept.sum())

        improved = cur_e < best_e
        best_state[improved] = state[improved]
        best_e[improved] = cur_e[improved]

    seqs = _decode(best_state)
    return SampleResult(
        config=config,
        sequences=[
            SeqRecord(seq=s, energy=float(sign * e), source=f"mcmc_python:{model.name}")
            for s, e in zip(seqs, best_e)
        ],
        accepted_fraction=accepted / (config.n_steps * n),
    )


def sample_sequences(
    config: SampleConfig,
    *,
    runner: Runner | None = None,
    prefer_julia: bool = False,
) -> SampleResult:
    """Generate sequences from a registered energy model.

    Routes to ``PhageMC.jl`` when ``prefer_julia`` is set and Julia is
    available, otherwise to the Python reference sampler.  Flip the default to
    True once ``agents/julia/sample.jl`` is wired to the real package.
    """
    entry = registry.get_model(config.model_name)

    if prefer_julia:
        runner = runner or default_runner()
        run_dir = ws.run_dir(ws.slug("adhoc-sample"), create=True)
        return runner.run("sample.jl", config, SampleResult, run_dir=run_dir, tag=config.model_name)

    if config.method != "mcmc":
        raise NotImplementedError(
            f"method={config.method!r} exists only on the Julia side. "
            "Set prefer_julia=True once agents/julia/sample.jl is wired up, "
            "or use method='mcmc' for the Python reference sampler."
        )
    model = load_energy_model(
        entry.weights_path, seq_length=entry.arch.seq_length, name=entry.name
    )
    return sample_mcmc_python(model, config)
