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
import re
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

#: Width of the one-hot encoding when the model was trained on gapped data.
#:
#: The campaign's models were: ``dense1_W`` in
#: ``data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json``
#: is 315x45 = (21*15) x (3*15).  The gap column is never occupied by a
#: designed binder -- ``colabdesign/energy_model/model_3layer.py`` pads a zero
#: column for it -- but the input width must still match or every number is
#: wrong.  Encoding width is therefore a property of the *loaded model*, not a
#: global constant.
_N_AA_GAP = _N_AA + 1

_DENSE_KEY = re.compile(r"^dense(\d+)_([Wb])$")


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

    # Layout D: {"dense1_W": ..., "dense1_b": ..., "dense2_W": ...}
    # This is what the campaign actually has: the layout written by
    # PD_energy_model/analysis/save_model_json.jl, by agents/julia/train.jl,
    # and read by colabdesign/energy_model/model_3layer.py.
    if isinstance(raw, dict) and any(_DENSE_KEY.match(k) for k in raw):
        idx = sorted(
            int(m.group(1)) for m in (_DENSE_KEY.match(k) for k in raw) if m and m.group(2) == "W"
        )
        out = []
        for i in idx:
            w = raw[f"dense{i}_W"]
            b = raw.get(f"dense{i}_b")
            out.append((_to_float_array(w), _to_float_array(b) if b is not None else np.zeros(1)))
        return out

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


def _infer_n_aa(first_w: np.ndarray, seq_length: int, arch: Any | None) -> int:
    """Decide whether the model was trained on 20 or 21 residue columns.

    Prefers the ``arch`` block written by ``agents/julia/train.jl``; otherwise
    matches the first layer's input width against both candidates.  Guessing
    wrong here is the failure mode the README warns about -- the forward pass
    would return confident nonsense -- so an ambiguous or unmatched width is an
    error, never a default.
    """
    if isinstance(arch, dict) and arch.get("n_aa"):
        n_aa = int(arch["n_aa"])
        if n_aa not in (_N_AA, _N_AA_GAP):
            raise ValueError(f"arch.n_aa must be {_N_AA} or {_N_AA_GAP}, got {n_aa}")
        return n_aa
    if first_w.ndim != 2:
        raise ValueError(f"first layer: expected a 2-D weight, got shape {first_w.shape}")
    widths = set(first_w.shape)
    matches = [n for n in (_N_AA, _N_AA_GAP) if seq_length * n in widths]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"first layer shape {first_w.shape} chains with neither orientation from an "
            f"input dim of {seq_length}*{_N_AA}={seq_length * _N_AA} or "
            f"{seq_length}*{_N_AA_GAP}={seq_length * _N_AA_GAP}. Check seq_length (got "
            f"{seq_length}) and the alphabet used at training time."
        )
    raise ValueError(
        f"first layer shape {first_w.shape} is ambiguous at seq_length={seq_length}: it "
        f"matches both a {_N_AA}- and a {_N_AA_GAP}-letter alphabet. Re-export the "
        "weights with an `arch` block (agents/julia/train.jl does) to disambiguate."
    )


@dataclass
class EnergyModel:
    """A loaded MLP energy model. Lower output = better binder, by convention."""

    layers: list[tuple[np.ndarray, np.ndarray]]
    activations: list[str]
    seq_length: int
    name: str = ""
    #: 20 (no gap) or 21 (trailing gap column), inferred from the weights.
    n_aa: int = _N_AA

    @property
    def input_dim(self) -> int:
        return self.seq_length * self.n_aa

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
    from the input width and raises if neither does.

    The input width is ``seq_length * n_aa`` where ``n_aa`` is 20 or 21,
    inferred from the first layer (and taken from the file's ``arch`` block
    when ``agents/julia/train.jl`` wrote one).  Both occur in this repository:
    the promoted models in ``data/energy_model_params/`` are 21-wide.

    Verified against ``colabdesign/energy_model/model_3layer.py``: both give
    9.0804653 for ``ACDEFGHIKLMNPQR`` under the promoted 2R weights.
    """
    p = Path(weights_path)
    if not p.is_absolute():
        p = ws.read_path(weights_path) if (ws.REPO_ROOT / weights_path).exists() \
            else ws.write_path(weights_path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    pairs = _extract_layers(raw)

    arch = raw.get("arch") if isinstance(raw, dict) else None
    if isinstance(arch, dict) and arch.get("seq_length"):
        seq_length = int(arch["seq_length"])
    n_aa = _infer_n_aa(pairs[0][0], seq_length, arch)

    fixed: list[tuple[np.ndarray, np.ndarray]] = []
    dim = seq_length * n_aa
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
        if isinstance(arch, dict) and arch.get("activations"):
            activations = [str(a) for a in arch["activations"]]
        else:
            # The campaign's convention, and the one model_3layer.py hard-codes:
            # relu hidden layers, linear output. Activations cannot be recovered
            # from weight matrices, so files without an `arch` block get this.
            activations = ["relu"] * (len(fixed) - 1) + ["identity"]
    if len(activations) != len(fixed):
        raise ValueError(f"{len(activations)} activations for {len(fixed)} layers")

    return EnergyModel(fixed, list(activations), seq_length, name or p.stem, n_aa)


# ==========================================================================
# Encoding
# ==========================================================================
def encode_onehot(
    sequences: Iterable[str], seq_length: int | None = None, n_aa: int = _N_AA
) -> np.ndarray:
    """One-hot encode sequences to ``(batch, L*n_aa)``, row-major over positions.

    ``n_aa=21`` appends an always-zero gap column per position, matching the
    gapped alphabet the campaign's models were trained on.  Pass
    ``model.n_aa``; do not assume 20.

    .. warning::
       The residue order is :data:`agents.core.schemas.AA_ALPHABET`, with the
       gap last. It must match the order used at training time in Julia
       (``PhageNegBinom.AMINO_ACIDS_GAP``).
    """
    if n_aa not in (_N_AA, _N_AA_GAP):
        raise ValueError(f"n_aa must be {_N_AA} or {_N_AA_GAP}, got {n_aa}")
    seqs = list(sequences)
    if not seqs:
        return np.zeros((0, 0))
    L = seq_length or len(seqs[0])
    out = np.zeros((len(seqs), L, n_aa), dtype=np.float64)
    for i, s in enumerate(seqs):
        if len(s) != L:
            raise ValueError(f"sequence {i} has length {len(s)}, expected {L}")
        for j, aa in enumerate(s):
            try:
                out[i, j, _AA_INDEX[aa]] = 1.0
            except KeyError:
                raise ValueError(f"sequence {i} position {j}: unknown residue {aa!r}") from None
    return out.reshape(len(seqs), L * n_aa)


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
    energies = model.energy(encode_onehot(sequences, model.seq_length, model.n_aa))
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
        return sign * model.energy(encode_onehot(_decode(mat), L, model.n_aa))

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

    ``method='gradient'`` and ``method='greedy'`` exist only on the Julia side
    and are routed there automatically -- ``agents/julia/sample.jl`` is wired
    to ``PhageMC.jl``.  ``method='mcmc'`` stays on the vectorised numpy path by
    default because it is the faster of the two and needs no Julia start-up;
    pass ``prefer_julia=True`` to run it through ``PhageMC.jl`` instead.
    """
    entry = registry.get_model(config.model_name)

    if prefer_julia or config.method != "mcmc":
        runner = runner or default_runner()
        run_dir = ws.run_dir(ws.slug("adhoc-sample"), create=True)
        return runner.run("sample.jl", config, SampleResult, run_dir=run_dir, tag=config.model_name)

    model = load_energy_model(
        entry.weights_path, seq_length=entry.arch.seq_length, name=entry.name
    )
    return sample_mcmc_python(model, config)
