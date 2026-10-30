# `agents/` — Agent system for the PA_PB1 campaign

A multi-agent layer over the existing binder-design pipeline. **Phase 1 covers
the fitness (energy) model only** — architecture, training, sequence
generation — and runs entirely on CPU. ColabDesign and cluster execution come
in phase 2, behind the `Runner` interface already defined in
`core/execution.py`.

Nothing outside this directory is modified. `PD_energy_model/`, `src/`,
`data/` and `results/` are read-only to the agents.

---

## Compartmentalisation

One writable root: **`agents/store/`**. Everything else in the repository is
readable but not writable.

The guarantee rests on three things, in order of importance:

1. **No agent has a generic filesystem or shell tool.** Only domain tools
   (`train_model`, `generate_sequences`, …). An agent cannot reach what no
   tool exposes. `tests/test_workspace.py` enforces this.
2. **Tools take identifiers, not paths.** `train_model(model_name=...)`, never
   `train_model(output_path=...)`. `workspace.slug()` validates the
   identifier; `workspace.write_path()` maps it to a location.
3. **A path guard.** `Path.resolve()` plus an `is_relative_to` assertion,
   which also catches symlinks planted inside the store.

Agent code itself (`core/`, `julia/`, `config/`) is **read-only at runtime** —
the agents cannot rewrite their own tools.

Deleting `agents/store/` returns the repository to its exact pre-agent state.

### Promotion is a human action

Models the agents train live in `agents/store/models/`. Copying one into
`data/energy_model_params/` — where the ColabDesign pipeline actually loads
it from — is deliberately **not** a tool. `registry.promotion_instructions(name)`
prints what to run. This is what stops an exploratory run from breaking the
design pipeline.

---

## Layout

```
agents/
├── core/
│   ├── workspace.py      path resolution + the write sandbox
│   ├── schemas.py        pydantic contracts (incl. the sign convention)
│   ├── execution.py      the Julia / SLURM boundary
│   ├── registry.py       agents/store/models/registry.json
│   ├── run.py            run directories and the audit trace
│   ├── tools/            plain Python functions — the only code touching data
│   │   ├── fitness.py       scoring (numpy fast path) + reference MCMC
│   │   ├── analysis.py      diversity, ranking, filtering, persistence
│   │   ├── architecture.py  what exists, what is valid, what it costs
│   │   └── training.py      training + evaluation via Julia
│   └── definitions/      LLM agents: system prompt + permitted tools
│       ├── base.py          Tool/Agent + the ONLY slapsci-aware function
│       ├── architect.py     decides the architecture
│       ├── trainer.py       trains and evaluates
│       ├── generator.py     samples and assesses candidates
│       └── orchestrator.py  the DBTL loop
├── julia/                CLI entry points → PhageNegBinom.jl / PhageMC.jl
├── config/default.yaml
├── store/                ← the only writable location
├── tests/
└── smoke_phase1.py       end-to-end check: no Julia, no GPU, no LLM
```

Layering is strictly downward: `definitions/` → `tools/` → `execution/registry/run` → `workspace/schemas`.
**Nothing below `definitions/` imports slapsci**, so the whole system is
testable with no provider and no API key.

---

## Quick start

```bash
pip install numpy pydantic pytest        # slapsci only when you wire the LLM loop
cd /path/to/PA_PB1

python -m pytest agents/tests -q         # 62 tests, no Julia needed
python -m agents.smoke_phase1            # full Generator loop on a synthetic model
```

The smoke script builds a model with *random* weights, registers it, generates
sequences, measures diversity, shortlists and saves — exercising the sandbox,
registry, loader, encoder, sampler, tool schemas and trace. It proves the
plumbing, not the science.

---

## The Julia boundary

The contract is process-based, not in-process:

```
config.json  →  julia --project=PD_energy_model agents/julia/<script>.jl  →  result.json
```

Chosen over `juliacall` for three reasons: the same contract becomes an
`sbatch` submission in phase 2 with no change above `execution.Runner`; a
crash or OOM in Julia does not take down the agent loop; and it matches the
JSON-result pattern the repo already uses.

If per-call latency becomes a problem, mitigate in this order: a
`PackageCompiler.jl` sysimage → a persistent Julia daemon → only then
`juliacall`.

### What still needs wiring

Three files, each with one clearly marked `BEGIN wire-up block`:

| File | Needs from the package |
|---|---|
| `julia/train.jl` | counts loader, model constructor, `train!`, weight exporter (`PhageNegBinom.jl`) |
| `julia/sample.jl` | the samplers — especially `gradient` and `greedy` (`PhageMC.jl`) |
| `julia/inspect.jl` | load a stored model, report real layer sizes |

`sample.jl` is **optional for phase 1**: `tools/fitness.py` has a pure-Python
reference MCMC. Flip `prefer_julia=True` in `sample_sequences()` once the Julia
sampler works.

---

## Two conventions that will silently corrupt results if broken

**Sign.** The model emits an *energy*: **lower is better**. This matches
`loss_total = loss_af + energy_weight * energy` in
`src/generative_protocols/model_energy_guidance.py`, which is minimised. Every
sampler minimises by default. If the Julia side returns a fitness where higher
is better, set `objective="maximize"` — do not flip the sign midway through
the stack.

**Alphabet.** `schemas.AA_ALPHABET` is `"ACDEFGHIKLMNPQRSTVWY"`. It must match
the one-hot order used when the Julia model was trained. If it does not,
`predict_fitness` returns confident nonsense rather than failing. Verify
before trusting any number from the Python fast path.

The loader does handle matrix orientation automatically — Julia stores
`Dense(in, out)` as `(out, in)` — by transposing whichever way chains from
`seq_length × 20`, and raising if neither does.

---

## First real validation (zero cost, no Julia)

Score the sequences with measured affinity and check the rank correlation:

```python
from agents.core.tools import training, analysis

seqs = analysis.read_seed_sequences("data/seqs_aff_alberto_darren.txt")
# load the matching measured values from
# data/affinity_measurements_PA-PB1_formatted.csv
training.evaluate_model("<registered-model>", seqs, measured)
```

This validates the weight loader, the alphabet ordering and the sign
convention in one shot — before any agent, any LLM and any GPU. Run it first.

An equally strong check: score the same sequences through
`model_energy_guidance.py` and confirm the two agree. They implement the same
forward pass from the same weights, so disagreement is a real bug.

---

## Connecting slapsci

`definitions/base.py::bind_to_slapsci` is the only slapsci-aware code. It is
written against the README's `get_llm_provider(provider, model)`, but the
keyword for registering tools is not documented there — check
`examples/tools_usage/example_tools_usage.py` in the slapsci repo and adjust
that one call. Tool schemas are already emitted in both OpenAI
(`to_openai_tools()`) and Anthropic (`to_anthropic_tools()`) shapes.

For the Architect agent, pass `ARCH_OUTPUT_MODEL` as slapsci's
`pydantic_model=` so its answer is a validated `ArchSpec` rather than prose to
parse.

Route every tool call through `AgentTeam.dispatch()` so the trace stays
complete.

---

## Phase 2 (not here yet)

- Implement `execution.SlurmRunner` — the signature is already fixed.
- Add `tools/generation.py` wrapping `src/generative_protocols/` (ColabDesign,
  AF2-multimer with the energy term in the loss).
- Add `tools/scoring.py` for Boltz-1 / AF3.
- Extend the orchestrator loop to feed structural scores back into selection.

Local ColabDesign testing on a single GPU: one AF2 model instead of five,
fewer recycles, short protocols, pTM instead of multimer for plumbing
smoke-tests.
