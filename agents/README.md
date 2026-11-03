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
cd /path/to/PA_PB1
uv sync                                  # or: pip install numpy pydantic pyyaml pytest
                                         # slapsci only when you wire the LLM loop

python -m pytest agents/tests -q         # 85 tests, no Julia needed
python -m agents.smoke_phase1            # full Generator loop on a synthetic model
```

**For task-by-task instructions and runnable examples, see
[USAGE.md](USAGE.md) and [examples/](examples/).** This README is the rationale:
why the system is shaped this way, and which conventions break it silently.

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

### What is wired

All three entry points are implemented against the packages:

| File | Uses |
|---|---|
| `julia/train.jl` | `PhageNegBinom.train!` / `log_likelihood` / `Node`, Flux chains, JSON export |
| `julia/sample.jl` | `SeqModelMC.Δenergy` and `descent!`, plus a Zygote gradient sampler |
| `julia/inspect.jl` | rebuilds the stored chain and reports its real layer sizes |

`julia/common.jl` holds what they share: the alphabet, one-hot encoding, chain
construction, and the weight export/load pair.

Two things about the packages are worth knowing before you read that code.

**`PhageNegBinom` is pinned to a branch.** `PD_energy_model/Manifest.toml`
resolves it to `changing_ll` of `SelectionScreeningNB.jl`, **not** the checkout
in `~/Projects/julia_packages/PhageNegBinom.jl`. The branch groups the
likelihood parameters into `data_params` / `distr_params` NamedTuples and calls
`regularizer(ps)` with the whole parameter tuple; the checkout uses positional
arguments and a splatted regulariser. `train.jl` calls the branch's form,
matching the training notebooks. If the Manifest is ever re-resolved to the
other version, `train.jl` is where it breaks.

**`PhageMC`'s module is `SeqModelMC`, and its samplers cannot take
constraints.** `anneal!` and `descent!` draw mutations with `rand(1:A)` over
the whole alphabet, so neither can honour `forbidden_aa` or `fixed_positions`
— and those are not optional, since cysteine is excluded by default.
`sample.jl` therefore builds its Metropolis and greedy loops on the package's
`Δenergy` primitive, and calls `descent!` directly only when there are no
constraints at all. `anneal_batch!` is unusable here: it reads `model.A` and
`model.L`, which a plain Flux `Chain` does not have. The `gradient` method has
no counterpart in the package and is implemented with Zygote (reached through
`Flux`, so no new dependency).

`sample_sequences()` now routes `gradient` and `greedy` to Julia
automatically. `mcmc` stays on the vectorised numpy path by default because it
is faster and needs no Julia start-up; pass `prefer_julia=True` to send it to
`PhageMC.jl` instead.

### The dataset manifest

`DatasetSpec` carries a single `counts_file`, but the experiment is two
replicate lineages of two rounds each. `train.jl` accepts either a wide CSV
(`Sequence` plus one count column per node) or a JSON manifest naming the
per-branch round files — `agents/config/dataset_2R.json` is the campaign's.
The manifest form is the only one that can express the tree.

Training follows the notebooks: one library state per branch, one shared
`selection` state (the energy model), a zeroed `wash` reference, and two
stages — libraries first, then selection. Fitting them jointly from scratch
lets the selection state absorb each library's composition, which is the exact
confound the energy model must not learn. Only `states.selection` is exported.

---

## Two conventions that will silently corrupt results if broken

**Sign.** The model emits an *energy*: **lower is better**. This matches
`loss_total = loss_af + energy_weight * energy` in
`src/generative_protocols/model_energy_guidance.py`, which is minimised. Every
sampler minimises by default. If the Julia side returns a fitness where higher
is better, set `objective="maximize"` — do not flip the sign midway through
the stack.

**Alphabet.** `schemas.AA_ALPHABET` is `"ACDEFGHIKLMNPQRSTVWY"` — but the
campaign's models are **21 residues wide**, not 20. `dense1_W` in
`data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json` is
315×45 = (21·15)×(3·15): the models were trained on
`PhageNegBinom.AMINO_ACIDS_GAP`, which appends the gap. A designed binder
never occupies that column — `model_3layer.py` pads a zero for it — but the
input width must still match.

Encoding width is therefore a property of the **loaded model**, not a global
constant: `EnergyModel.n_aa` is inferred from the first layer (or read from
the `arch` block `train.jl` writes), and `encode_onehot` takes it as an
argument. Pass `model.n_aa`; never assume 20. An ambiguous width is an error,
not a default.

The loader also handles matrix orientation — Julia stores `Dense(in, out)` as
`(out, in)` and serialises column-major, so the file arrives as `(in, out)`
and the forward pass is `x @ W`, not `x @ W.T`. It transposes whichever way
chains from the input width and raises if neither does.

**Activations.** They are not recoverable from weight matrices. Files written
by `train.jl` record them in the `arch` block; for older files the loader
falls back to the campaign's convention — relu hidden, identity output —
which is what `model_3layer.py` hard-codes.

---

## Verified agreement across the three forward passes

The same weights, scored three ways — the Flux chain `common.jl` rebuilds, the
numpy fast path in `tools/fitness.py`, and `model_3layer.py` as ColabDesign
calls it — agree to float32 precision. For `ACDEFGHIKLMNPQR` under the
promoted 2R weights all three give **9.080465**. `test_fitness.py` pins that
value, so the alphabet, orientation and activation conventions cannot drift
apart silently.

A model trained by `train.jl` is a drop-in for the pipeline: its exported file
loads unchanged in `model_3layer.py` and produces identical numbers. Promotion
is still a human action — see `registry.promotion_instructions(name)`.

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
