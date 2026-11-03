# `agents/` — practical guide

How to use the agent layer. For *why* it is built this way — the
compartmentalisation argument, the Julia boundary, the conventions that will
silently corrupt results if broken — see [README.md](README.md).

---

## 1. What this is

The PA–PB1 campaign designs 15-residue peptides that bind the influenza
polymerase PA subunit. Two models drive it: **AlphaFold2**, via ColabDesign,
proposes sequences that fold against the target, and an **MLP energy model**,
trained on phage-display sequencing counts with `PhageNegBinom.jl`, predicts
how well a sequence binds. The energy model enters the design loop as a term in
the ColabDesign loss:

```
loss_total = loss_af + energy_weight * energy
```

This directory adds a layer on top of that energy model: tools for deciding an
architecture, training it, sampling candidate binders from it, and measuring
whether those candidates are any good — each call validated against a schema
and recorded in an audit trace. Three agents hold the tools, split by role
(architect, trainer, generator), so an LLM can be dropped in to choose the
arguments. **No LLM is required**: everything here runs by calling the tools
directly, which is how the tests and the examples work.

**What it does not do yet.** No ColabDesign, no AF2, no GPU, no cluster. The
loop stops at "here are candidate sequences with predicted energies". Feeding
those into structure prediction is phase 2, behind the `Runner` interface in
[`core/execution.py`](core/execution.py).

---

## 2. Setup

The agent layer needs far less than the rest of the repository: **no `.env`,
no CUDA, no AF2 weights**. Julia is needed only for training, Julia-side
sampling and model inspection.

```bash
cd /path/to/PA_PB1
uv sync                       # or: pip install numpy pydantic pyyaml pytest
```

Check it works. Neither command needs Julia:

```bash
python -m pytest agents/tests -q      # 85 passed
python -m agents.smoke_phase1         # ends: OK -- phase-1 plumbing works end to end.
```

Then run the first example, which scores real peptides with the model the
production pipeline uses:

```bash
PYTHONPATH=. python agents/examples/01_score_promoted_model.py
```

**`PYTHONPATH=.` and the repository root.** Everything is imported as
`agents.*`, so commands run from the repository root. Scripts under
`agents/examples/` are invoked by path with `PYTHONPATH=.`; `smoke_phase1.py`
is invoked as a module (`python -m agents.smoke_phase1`). Either form works, as
long as the root is on the path.

---

## 3. Five things to know first

**`agents/store/` is the only writable location.** Trained models, generated
sequences and run traces all land there. Nothing else in the repository is
written by this code. Deleting `agents/store/` returns the repository to its
exact pre-agent state — treat it as disposable scratch space.

**Tools take identifiers, not paths.** `train_model(model_name=...)`, never
`train_model(output_path=...)`. There is deliberately no filesystem or shell
tool anywhere in the package: an agent cannot reach what no tool exposes.

**Promotion is a human action.** Copying a trained model into
`data/energy_model_params/`, where ColabDesign loads it from, is not a tool.
`registry.promotion_instructions(name)` prints the `cp` to run. This is what
stops an exploratory run from quietly changing the design pipeline.

**Lower energy is better.** The model is a cost, not a score. Every sampler
minimises by default. If something ever returns a fitness where higher is
better, set `objective="maximize"` rather than flipping a sign midway through
the stack.

**The alphabet is 21 wide, not 20.** The campaign's models were trained on the
gapped alphabet — 20 residues plus a trailing gap column — so `dense1_W` is
315×45, not 300×45. A designed binder never occupies the gap column, but the
input width must match or the forward pass returns confident nonsense. Encoding
width therefore belongs to the **loaded model**, not to a constant:

```python
model = fitness.load_energy_model(path, seq_length=15)
x = fitness.encode_onehot(seqs, 15, model.n_aa)     # pass model.n_aa
```

The same goes for `ArchSpec.n_aa`, which defaults to 21.

---

## 4. Three ways to call the same thing

| | How | Validated | Traced | Use when |
|---|---|---|---|---|
| Plain function | `fitness.predict_fitness(seqs, "m")` | no | no | scripting, notebooks, exploration |
| Agent tool | `agent.call_tool("predict_fitness", ...)` | yes | no | checking what an LLM would be allowed to do |
| Team dispatch | `team.dispatch("generator", "predict_fitness", ...)` | yes | yes | anything whose provenance matters |

Two sharp edges:

- The `@tool` decorator **replaces** the function with a `Tool` object. So
  `generator.predict_fitness` is not callable — use
  `generator.predict_fitness.call(**kwargs)`, or go through `call_tool` /
  `dispatch`.
- `call_tool` and `dispatch` take their leading arguments **positionally**
  (`dispatch("generator", "predict_fitness", ...)`), because several tools have
  their own `name` and `model_name` parameters that would otherwise collide.

---

## 5. Task recipes

Each one has a runnable script in [`examples/`](examples/).

### Score sequences — no Julia

```bash
PYTHONPATH=. python agents/examples/01_score_promoted_model.py
```

```python
from agents.core.tools import fitness, analysis

model = fitness.load_energy_model(
    "data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json",
    seq_length=15,
)
seqs = analysis.read_seed_sequences("data/seqs_aff_alberto_darren.txt")
energies = model.energy(fitness.encode_onehot(seqs, 15, model.n_aa))
```

For a model that is in the registry, `fitness.predict_fitness(seqs, "model-name")`
does the loading and encoding for you.

### Validate against measured affinity — no Julia

The cheapest meaningful check in the whole system. It exercises the weight
loader, the alphabet ordering and the sign convention at once, against measured
numbers rather than the model's own output.

```bash
PYTHONPATH=. python agents/examples/02_validate_against_affinity.py
```

```
35 measured sequences from data/affinity_measurements_PA-PB1_formatted.csv
Spearman (energy vs Kd): +0.4522
Pearson  (energy vs Kd): +0.5827
```

**Reading the sign.** Low Kd = strong binder; low energy = good binder. They
run in the same direction, so a **positive** Spearman is the healthy outcome. A
negative one is a sign-convention bug, not a weak model. (These peptides
overlap the training data, so this measures wiring, not generalisation.)

### Train a model — needs Julia

```bash
PYTHONPATH=. python agents/examples/03_train_model.py --name my-model --iters 20000
```

The dataset is named by a **manifest**, not a single file: the experiment is
two replicate lineages (F, R) of two rounds each, and `DatasetSpec` carries one
path. [`config/dataset_2R.json`](config/dataset_2R.json) is the campaign's.
`train.jl` also accepts a wide CSV (`Sequence` plus one count column per node)
for small or synthetic data.

Training follows the campaign's notebooks: one library state per branch, one
shared `selection` state — the energy model — a zeroed `wash` reference, and
two stages, libraries first and selection second. Only `states.selection` is
exported. Weights land in `agents/store/models/<name>/`, and the file is a
drop-in for `model_3layer.py`.

The default `--iters 40` finishes in about 40 s and is a plumbing check, not a
trained model. Raise it by two or three orders of magnitude for real work.

### Sample candidates — Julia for two of the three methods

```bash
PYTHONPATH=. python agents/examples/04_sample_candidates.py --model my-model --compare
```

| Method | Runs on | Character |
|---|---|---|
| `mcmc` | numpy (Julia with `prefer_julia=True`) | robust default; most diverse |
| `greedy` | Julia | lowest energy; collapses onto a motif |
| `gradient` | Julia | descent on soft logits; collapses hardest |

`greedy` and `gradient` route to `agents/julia/sample.jl` automatically.
Constraints (`forbidden_aa`, `fixed_positions`) are honoured by all three.

The tradeoff is real and worth seeing before choosing:

```
--- mcmc      mean -0.3449  best -0.5215   clusters 24   entropy 3.295 bits
--- greedy    mean -1.2339  best -1.3574   clusters 15   entropy 1.225 bits
--- gradient  mean -0.8639  best -0.9376   clusters  4   entropy 0.745 bits
```

Energy alone is not success. A sampler optimising energy will collapse onto one
motif, and a shortlist of near-identical sequences is one experiment, not
twenty — always read `check_diversity` alongside the energies.

### Shortlist and save

```python
from agents.core.tools import analysis
from agents.core.schemas import SeqConstraints

kept = analysis.filter_sequences(records, SeqConstraints(forbidden_aa="C"))
top = analysis.rank_sequences(kept, top_k=10)
analysis.save_sequences(top, "my-shortlist")     # store/sequences/my-shortlist.{json,csv}
```

`load_sequences(name)` reads the JSON back; the CSV is for humans.

### Inspect, compare, validate

```python
from agents.core.tools import architecture

architecture.list_trained_models()                     # what the agents have trained
architecture.list_reference_models()                   # what is in data/energy_model_params/
architecture.inspect_model("my-model", use_julia=True) # ground truth from the stored weights
architecture.compare_models(["my-model", "other"])
```

`inspect_model(use_julia=True)` rebuilds the model in Flux and reports the
layer sizes and width it actually finds, warning when they disagree with the
`ArchSpec`. That disagreement is the bug class worth catching early.

### Promote a model — human only

```python
from agents.core import registry
print(registry.promotion_instructions("my-model"))
```

It prints the `cp` and tells you to repoint `ENERGY_WEIGHTS_PATH`. Run it
yourself; no tool will.

### Run the whole loop with the trace

```bash
PYTHONPATH=. python agents/examples/05_full_loop.py --model my-model
```

Shows `build_team()`, `team.dispatch(...)` inside `with team.run:`, the summary,
and the resulting `trace.jsonl`.

### Connect an LLM

`definitions/base.py::bind_to_slapsci` is the only provider-aware code. One
detail is still unverified: the keyword for registering tools is not documented
in the slapsci README — check `examples/tools_usage/example_tools_usage.py` in
that repo and adjust the single `tools=` call. Tool schemas are already emitted
in both OpenAI (`to_openai_tools()`) and Anthropic (`to_anthropic_tools()`)
shapes. For the architect, pass `ARCH_OUTPUT_MODEL` as slapsci's
`pydantic_model=` so its answer is a validated `ArchSpec` rather than prose.
Route every call through `AgentTeam.dispatch()` so the trace stays complete.

---

## 6. Reference

### Store layout

```
agents/store/
├── models/
│   ├── registry.json              what was trained, with metrics  (tracked in git)
│   └── <name>/<name>_energy_model_weights.json
├── sequences/<name>.{json,csv}    saved candidate sets
├── datasets/
└── runs/<run_id>/
    ├── manifest.json              run_id, goal, status, started, duration_s
    ├── trace.jsonl                one JSON object per tool call
    ├── summary.md                 if written
    └── artifacts/                 config.json / result.json / logs per Julia call
```

A run id looks like `20261030T142210Z_example-loop`. A `trace.jsonl` record is
`{ts, tool, params, result}` plus `error` when a call raised; long values are
truncated, so the trace stays readable but is not a complete data record.

### Tools by agent

| Agent | Tools |
|---|---|
| architect | `list_trained_models`, `list_reference_models`, `inspect_model`, `validate_architecture`, `compare_models`, `describe_available_data` |
| trainer | `describe_available_data`, `prepare_dataset`, `estimate_cost`, `list_trained_models`, `train_model`, `evaluate_model`, `how_to_promote` |
| generator | `predict_fitness`, `generate_sequences`, `check_diversity`, `rank_candidates`, `save_candidates`, `load_reference_sequences` |

The architect cannot train. The trainer cannot promote. The generator cannot
train. That is the point of the split.

### Configuration

[`config/default.yaml`](config/default.yaml) is read at import by
[`core/config.py`](core/config.py); its values become the defaults in the tool
schemas the LLM sees. Editing it changes behaviour on the next run.

| Key | Used by |
|---|---|
| `llm.provider`, `llm.model`, `llm.per_agent.*` | `bind_to_slapsci`, `build_*_agent` |
| `execution.julia_project`, `julia_bin`, `timeout_s` | `LocalJuliaRunner` |
| `execution.backend` | `default_runner()` (`local`; `slurm` is phase 2) |
| `defaults.seq_length`, `defaults.forbidden_aa` | `generate_sequences` |
| `sampling.n`, `temperature`, `n_steps` | `generate_sequences` |
| `sampling.min_mean_hamming` | `check_diversity`'s collapse flag |
| `data.*` | informational; no tool reads these yet |

`DEFAULTS` in `core/config.py` mirrors the file and is the merge base, so a
missing or partial file degrades to the built-in values. Set
`PA_PB1_AGENTS_CONFIG` to use a different file.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'agents'` | not at the repository root | `cd` to the root and prefix `PYTHONPATH=.` |
| `chains with neither orientation` | wrong `seq_length`, or weights from a different alphabet | check the first layer's width against `seq_length × 20` and `× 21` |
| `model 'x' already exists` | the registry keeps comparison history | pick a new name, or pass `overwrite=True` |
| `backend produced no result file` | Julia failed before writing `result.json` | read `artifacts/*.log` in the run directory; the full stderr is there |
| `Package Zygote not found` | a Julia script imported an indirect dependency | reach it through `Flux`; do not edit `PD_energy_model/Project.toml` |
| First Julia call takes minutes | precompilation | expected once per environment; later calls are ~40 s |
| `CUDA_ERROR_NO_DEVICE` noise from JAX | no GPU on this machine | harmless, or set `JAX_PLATFORMS=cpu` |
| `no such file in the repository` | `read_seed_sequences` takes repo-relative paths | drop any leading `/` and `..` |

---

## 8. Known limits

- **Poisson likelihood is untested.** `train.jl` supports it, but only
  `negbinom` has been run.
- **`SlurmRunner` is a stub.** Its signature is fixed; setting
  `execution.backend: slurm` raises a clear error rather than pretending.
- **`load_sequences` only reads the JSON half** of `save_sequences`, so
  `fmt="csv"` is not round-trippable.
- **`read_seed_sequences` cannot parse the affinity CSV.** It takes the first
  whitespace token per line, which yields header text. Use `csv.DictReader`, as
  example 02 does.
- **`bind_to_slapsci`'s tool keyword is unverified** — see §5.
- **`PhageMC`'s own samplers cannot honour constraints**, so `sample.jl` builds
  its Metropolis and greedy loops on the package's `Δenergy` primitive. See
  [README.md](README.md) for the detail.
