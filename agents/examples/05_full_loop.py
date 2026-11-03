"""
05_full_loop.py

The same work as examples 01-04, but routed through the agent team.

This is what the agent layer adds over calling the tools directly: every call
is validated against its schema and appended to an audit trace, so a run is
reproducible and reviewable after the fact. No LLM is involved -- an LLM would
choose the arguments, and here the script does. Everything below the agent
definitions is provider-independent, which is why this runs with no API key.

Three agents, each holding only the tools its role needs:

    architect   decides the architecture; cannot train or sample
    trainer     trains and evaluates; cannot promote
    generator   samples and assesses candidates

Usage:
    PYTHONPATH=. python agents/examples/05_full_loop.py                  # uses an existing model
    PYTHONPATH=. python agents/examples/05_full_loop.py --train          # trains a fresh one first
    PYTHONPATH=. python agents/examples/05_full_loop.py --model doc-demo
"""

import argparse
import json
import time

import numpy as np

from agents.core import registry
from agents.core.definitions import build_team

MANIFEST = "agents/config/dataset_2R.json"
KNOWN_BINDERS = "data/seqs_aff_alberto_darren.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the whole loop through the agent team.")
    ap.add_argument("--model", default="doc-demo", help="model to sample from")
    ap.add_argument("--train", action="store_true", help="train a fresh model first (slow)")
    ap.add_argument("--iters", type=int, default=40, help="training iterations if --train")
    ap.add_argument("--n", type=int, default=24, help="sequences to generate")
    args = ap.parse_args()

    team = build_team(goal="example end-to-end loop", label="example-loop")
    print(f"run directory: {team.run.dir}")
    for name, info in team.describe().items():
        print(f"  {name:10s} {len(info['tools'])} tools: {', '.join(info['tools'])}")

    model_name = args.model

    # Everything inside the `with` is traced. The manifest records the run as
    # failed if an exception escapes, so a crashed run is distinguishable from
    # one that was never finished.
    with team.run:
        # -- architect: what exists, and is the design sane? ------------------
        existing = team.dispatch("architect", "list_trained_models")
        print(f"\n[architect] {len(existing)} model(s) already trained")

        check = team.dispatch(
            "architect", "validate_architecture",
            name="loop-demo", layer_units=[45, 15, 1],
            layer_activations=["relu", "relu", "identity"],
            n_training_examples=26000,
        )
        print(f"[architect] input_dim={check['input_dim']} n_params={check['n_params']}")
        for w in check["warnings"]:
            print(f"[architect] warning: {w}")

        # -- trainer: data, cost, and optionally a fresh model ----------------
        data = team.dispatch("trainer", "describe_available_data")
        print(f"[trainer] {len(data['data'])} files under data/, "
              f"{len(data['energy_model'])} under PD_energy_model/data/")

        if args.train:
            model_name = f"loop-{time.strftime('%H%M%S')}"
            print(f"[trainer] training {model_name} ({args.iters} iters) -- calls Julia")
            trained = team.dispatch(
                "trainer", "train_model",
                model_name=model_name, layer_units=[45, 15, 1],
                layer_activations=["relu", "relu", "identity"],
                counts_file=MANIFEST, l2=1e-2, epochs=args.iters,
                learning_rate=1e-4, batch_size=256,
                notes="trained by 05_full_loop.py",
            )
            print(f"[trainer] val_loss={trained['metrics']['val_loss']:,.2f} "
                  f"n_params={trained['n_params']}")
        else:
            registry.get_model(model_name)   # fail early and clearly if absent
            print(f"[trainer] skipped; sampling from the existing {model_name!r}")

        # -- generator: baseline, generate, assess, shortlist, save -----------
        known = team.dispatch("generator", "load_reference_sequences",
                              repo_relative_path=KNOWN_BINDERS)
        scored = team.dispatch("generator", "predict_fitness",
                               sequences=known[:8], model_name=model_name)
        print(f"\n[generator] known binders scored: "
              f"mean {np.mean([r['energy'] for r in scored]):+.4f}")

        gen = team.dispatch("generator", "generate_sequences",
                            model_name=model_name, n=args.n, length=15,
                            temperature=0.5, n_steps=400, seed=1)
        seqs = [r["seq"] for r in gen["sequences"]]
        energies = [r["energy"] for r in gen["sequences"]]
        print(f"[generator] generated {len(seqs)}  mean {np.mean(energies):+.4f}  "
              f"best {min(energies):+.4f}  acceptance {gen['accepted_fraction']:.3f}")

        div = team.dispatch("generator", "check_diversity", sequences=seqs)
        print(f"[generator] clusters {div['n_clusters']}  "
              f"mean_hamming {div['mean_hamming']}  collapsed={div['collapsed']}")

        top = team.dispatch("generator", "rank_candidates",
                            sequences=seqs, model_name=model_name, top_k=5)
        print("[generator] top 5:")
        for r in top:
            print(f"    {r['seq']}  {r['energy']:+.4f}")

        saved = team.dispatch("generator", "save_candidates",
                              sequences=[r["seq"] for r in top],
                              energies=[r["energy"] for r in top],
                              name="loop-candidates", source=model_name)
        print(f"[generator] saved {saved['json']}")

        team.run.write_summary(
            f"# Example loop\n\n"
            f"- model: {model_name}\n"
            f"- generated: {len(seqs)}\n"
            f"- acceptance: {gen['accepted_fraction']:.3f}\n"
            f"- mean energy: {np.mean(energies):+.4f}\n"
            f"- clusters: {div['n_clusters']}\n"
        )

    # -- the trace is the point --------------------------------------------
    trace = (team.run.dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    print(f"\ntrace: {len(trace)} calls recorded")
    for line in trace:
        rec = json.loads(line)
        print(f"  {rec['ts'][11:19]}  {rec['tool']}")
    print(f"\nmanifest: {json.loads((team.run.dir / 'manifest.json').read_text())['status']}")
    print(f"artefacts in {team.run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
