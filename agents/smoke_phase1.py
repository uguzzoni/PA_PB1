"""Phase-1 smoke test: the Generator loop end to end, with no Julia, no GPU, no LLM.

Builds a synthetic energy model with random weights, registers it, and drives
the Generator agent's tools exactly as the LLM would -- through
``AgentTeam.dispatch``, so the run trace is produced too.

Run from the repository root::

    python -m agents.smoke_phase1

What it proves: the workspace sandbox, the registry, the weight loader, the
one-hot encoding, the MCMC sampler, the diversity metrics, the tool schemas
and the run trace all work together. What it does *not* prove: that the
energy model is any good -- the weights are random noise.

Once ``agents/julia/train.jl`` is wired up, replace the synthetic model with a
real one and the same script becomes a genuine regression test.
"""

from __future__ import annotations

import json

import numpy as np

from .core import registry, workspace as ws
from .core.definitions import build_team
from .core.schemas import ArchSpec, EvalMetrics, Layer, ModelEntry

SYNTHETIC = "synthetic-demo"


def make_synthetic_model(name: str = SYNTHETIC, seq_length: int = 15) -> ModelEntry:
    """A registered model with random weights, in the expected export layout."""
    units = [16, 8, 1]
    acts = ["tanh", "tanh", "identity"]
    arch = ArchSpec(
        name=name,
        layers=[Layer(units=u, activation=a) for u, a in zip(units, acts)],
        seq_length=seq_length,
        notes="Random weights. For plumbing tests only -- the energies are meaningless.",
    )

    rng = np.random.default_rng(0)
    # 21-wide, matching the gapped alphabet the campaign's real models use,
    # so the synthetic weights agree with the ArchSpec describing them.
    dims = [seq_length * 21] + units
    layers = [
        {
            "W": (rng.normal(size=(a, b)) * 0.1).tolist(),
            "b": rng.normal(size=b).tolist(),
        }
        for a, b in zip(dims, dims[1:])
    ]

    ws.ensure_store()
    mdir = ws.model_dir(name, create=True)
    wfile = mdir / f"{name}_energy_model_weights.json"
    wfile.write_text(json.dumps({"layers": layers}), encoding="utf-8")

    return registry.register_model(
        ModelEntry(
            name=name,
            arch=arch,
            metrics=EvalMetrics(n=0),
            weights_path=str(wfile.relative_to(ws.STORE)),
            n_params=arch.n_params(),
        ),
        overwrite=True,
    )


def main() -> int:
    entry = make_synthetic_model()
    team = build_team(goal="phase-1 smoke test", label="smoke")

    print(f"run directory: {team.run.dir}")
    print(f"model: {entry.name}  ({entry.n_params} params)\n")

    with team.run:
        # 1. Baseline: score a few arbitrary sequences.
        baseline_seqs = ["ACDEFGHIKLMNPQR", "MKTAYIAKQRQISFV"[:15], "WWWWWWWWWWWWWWW"]
        baseline = team.dispatch(
            "generator", "predict_fitness", sequences=baseline_seqs, model_name=entry.name
        )
        print("baseline energies:")
        for row in baseline:
            print(f"  {row['seq']}  {row['energy']:+.4f}")

        # 2. Generate.
        gen = team.dispatch(
            "generator",
            "generate_sequences",
            model_name=entry.name,
            n=30,
            length=15,
            temperature=0.8,
            n_steps=400,
            seed=0,
        )
        seqs = [r["seq"] for r in gen["sequences"]]
        energies = [r["energy"] for r in gen["sequences"]]
        print(f"\ngenerated {len(seqs)} sequences")
        print(f"  acceptance : {gen['accepted_fraction']:.3f}")
        print(f"  mean energy: {np.mean(energies):+.4f}")
        print(f"  best energy: {np.min(energies):+.4f}")

        # 3. Diversity -- the check that stops a collapsed run looking like success.
        div = team.dispatch("generator", "check_diversity", sequences=seqs)
        print("\ndiversity:")
        for k, v in div.items():
            print(f"  {k:24s} {v}")
        if div["n_clusters"] <= 1 or div["mean_hamming"] < 3:
            print("  !! chains collapsed -- raise temperature or n_steps")

        # 4. Shortlist and save.
        top = team.dispatch(
            "generator", "rank_candidates", sequences=seqs, model_name=entry.name, top_k=5
        )
        print("\ntop 5:")
        for r in top:
            print(f"  {r['seq']}  {r['energy']:+.4f}")

        paths = team.dispatch(
            "generator",
            "save_candidates",
            sequences=[r["seq"] for r in top],
            energies=[r["energy"] for r in top],
            name="smoke-candidates",
            source=entry.name,
        )
        print(f"\nsaved: {paths}")

        team.run.write_summary(
            f"# Smoke run\n\nModel: {entry.name} (random weights)\n\n"
            f"- generated: {len(seqs)}\n- acceptance: {gen['accepted_fraction']:.3f}\n"
            f"- mean energy: {np.mean(energies):+.4f}\n- clusters: {div['n_clusters']}\n"
        )

    trace = (team.run.dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    print(f"\ntrace: {len(trace)} tool calls recorded in {team.run.dir / 'trace.jsonl'}")
    print("\nOK -- phase-1 plumbing works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
