# Handoff: the evaluation and objective layer

Two new modules were added to `agents/core/`, with tests and two examples.
This note explains what they are for, how to integrate them, and — most
importantly — **which decisions in them look like mistakes but are not**.

Read [README.md](README.md) for the original architecture and
[USAGE.md](USAGE.md) for day-to-day use.

---

## 1. This is additive

Six new files. Nothing existing was modified:

| File | What |
|---|---|
| `core/evaluation.py` | metrics with bootstrap CIs; enrichment; disagreement ranking |
| `core/objective.py` | tiered objective, saturation rule, campaign state, verdicts |
| `tests/test_evaluation.py` | statistics, orientation, disagreement |
| `tests/test_objective.py` | the protocol: saturation, demotion, gates, verdicts |
| `examples/06_campaign_protocol.py` | the protocol end to end, synthetic |
| `examples/07_next_measurements.py` | choosing which sequences to measure next |

Neither module imports `schemas.py`, `config.py`, `tools/` or anything that
touches sequence encoding, so the gapped 21-wide alphabet and the rest of the
campaign-specific wiring are irrelevant to them. `evaluation.py` takes
**arrays of predicted energies**, never a model. That decoupling is
deliberate: it keeps the statistics testable without a trained model.

```bash
python -m pytest agents/tests -q      # 152 passed (85 existing + 67 new)
PYTHONPATH=. python agents/examples/06_campaign_protocol.py
PYTHONPATH=. python agents/examples/07_next_measurements.py
```

---

## 2. What problem these solve

The goal is an agent system that *improves* the energy model across three
axes — architecture, PhageNegBinom metaparameters, training procedure —
judged by several metric families (affinity correlation, enrichment
agreement, held-out count fit).

The failure mode that makes this hard is not computational. It is that an
LLM free to pick which metric to look at **after** seeing the results will
always find a story: "affinity dropped but enrichment improved, so this is
better". Over fifty experiments that produces a model that looks improved and
is not. These modules exist so that *"is this better?"* is a **computed**
answer, not a narrated one.

### The tiered objective

Tiers are declared once, in order. Each is optimised until it saturates, then
**demoted to a frozen gate** while the next takes over:

```
tier 1   affinity.KD.spearman      optimise, then demote to gate
tier 2   enrichment.spearman       optimise, then demote to gate
tier 3   counts.heldout_loglik     final tie-break
```

Affinity goes first because early on, when models differ a lot, ~35 measured
peptides can still tell them apart. Once they cannot — intervals all
overlapping — it demotes to a gate and enrichment, which has orders of
magnitude more data, takes over.

The role reversal at demotion is the point: **before demotion affinity can
only help a candidate** (as a tie-break); **after demotion it can only block
one** (as a gate).

---

## 3. Integration tasks

In order. None is urgent enough to block running the examples.

**(a) Measure the real noise floor.** Nothing else is meaningful first. Train
the *same* spec with 5 seeds, build an `EvaluationReport` for each, then:

```python
from agents.core import objective as obj
rep = obj.summarise_replication("baseline", reports)
floors = obj.noise_floor_from_replication(rep, k=2.0)
```

Those floors go into the `ObjectiveSpec`. Until they are measured from real
runs, the saturation rule is calibrated on invented numbers and the whole
protocol turns over without engaging.

**(b) Reconcile the two metric types.** There are now two in the codebase:

- `schemas.EvalMetrics` — point estimates, returned by `tools/training.evaluate_model`
- `evaluation.MetricWithCI` / `EvaluationReport` — with bootstrap intervals

`EvalMetrics` cannot support the protocol: a point estimate is not comparable
to another point estimate when n≈35. `evaluate_model` should be migrated to
return an `EvaluationReport`. Keep `EvalMetrics` only if something else still
depends on it.

**(c) Wire a real report builder.** `evaluation.build_report()` takes
predicted energies plus measurements. Something must feed it: score the
model over the measured-affinity set and over the counts table, then call it.
This is the one piece that needs the campaign's actual data layout (the F/R
branches, the round structure) and so could not be written blind.

Note the enrichment definition is **not** fixed by these modules — they take
`counts_before` / `counts_after` arrays. For the F/R two-round design, decide
what those mean (round₂/round₁, final/library, or a fitted slope) and keep it
constant across the campaign.

**(d) Consider tracking campaign state in git.** `agents/store/campaigns/*.json`
holds the objective, the full history and the demotion log. It is currently
caught by `store/*` in `.gitignore`. It is small and is the record of why the
search went the way it did — arguably it deserves the same exception
`registry.json` has.

**(e) Then the Proposer agent.** Reads the campaign history, proposes the next
`ExperimentSpec` with a **pre-registered hypothesis and predicted direction**,
which `Campaign.record()` already accepts. `TrainingSpec` and
`LikelihoodSpec` need writing to make axes 1 and 2 concrete.

---

## 4. Decisions that must not be "simplified"

Each of these looks wrong at a glance. Each is load-bearing, and each has a
test naming it. **Do not relax one without reading its test first.**

**The gate rejects only *provable* harm**, not a lower point estimate:
`ci_high < reference - tolerance`. Rejecting on point estimates would discard
good models at random, because with ~35 points the interval is ±0.3 wide.
A gate that fires on noise is worse than no gate.
→ `test_noisily_worse_affinity_gets_the_benefit_of_the_doubt`

**Improvement requires clearing the incumbent's point estimate with the
candidate's CI low**, not `point > point`. The difference between 0.45 and
0.50 on 35 observations does not exist.
→ `test_improvement_requires_clearing_the_interval`

**Tier demotion is irreversible and persisted.** There is deliberately no
method to go back. Reversibility is cherry-picking with extra steps: a
campaign that can return to whichever metric is currently flattering it has
no protocol at all.
→ `test_switch_survives_reload_and_cannot_be_undone`

**The gate threshold freezes once and never re-freezes.** Letting it track
the best-so-far would make it drift upward as sampling continues, which is a
selection effect in its own right.
→ `test_demoted_threshold_never_re_freezes`

**Saturation is a measured condition, never an agent's judgement.** An agent
allowed to declare saturation will declare it whenever it is losing. The rule
needs all three: enough experiments, top-k intervals mutually overlapping,
and no improvement beyond the noise floor within the patience window.
→ `TestSaturation` (four cases, three of them negative)

**`enrichment.spearman` is the objective; the Pearsons are diagnostics.**
Enrichment is compressed at both ends — a sequence cannot exceed the whole
library nor fall below zero counts — so its relation to energy is monotone
but nonlinear in the tails, and Pearson is misspecified there. Ranks also
bound each observation's leverage, which contains the heteroscedasticity
without modelling it. Do not "upgrade" the objective to the weighted Pearson:
that needs a correct dispersion parameter, and the NB assumption, and the
delta method, all to be right.
→ `test_spearman_survives_tail_compression`

**A tied experiment is not a failed prediction** in the hypothesis scorecard.
An experiment that could not tell does not refute the hypothesis; scoring
ties as misses would understate the proposer systematically.
→ `test_ties_are_inconclusive_not_misses`

**`kon` is oriented opposite to `KD`.** A high kon is a good binder; a low Kd
is a good binder. Both end up positive after orientation, and the raw sign is
kept for diagnosis. Getting this backwards inverts a metric silently.
→ `test_kon_is_flipped`

---

## 5. Two things worth knowing

**Why affinity is a gate rather than the permanent objective.** Enrichment is
a *proxy* for the physical quantity selection acts on. A model's estimate
pools information across sequences, so it is often *less* noisy than a single
sequence's enrichment — which means a more flexible model that partly fits
the enrichment noise can correlate better with observed enrichment than a
smoother model that is closer to the truth. Optimising a noisy proxy hard
selects for noise-fitting. The affinity gate is the specific defence against
that: it catches a model that won on enrichment by learning its noise but
lost on the physical quantity.

**Where to spend new affinity measurements.** Follows from the same premise.
`evaluation.rank_by_disagreement()` ranks sequences by rank-gap between model
and enrichment, filtered for depth and spread by Hamming distance. In the
worked example the overlap with "just measure the top predicted binders" is
**0 of 8**: the top candidates are exactly where model and data already
agree, so measuring them confirms rather than arbitrates. Each disagreement
measured resolves a real conflict *and* adds a point to the affinity set,
which is the scarce resource the gate depends on.

---

## 6. Deliberately absent

- **No LLM computes a metric.** `evaluation.py` is pure and deterministic.
  Evaluation is not an agent, and should not become one — an LLM there is a
  place for narrative to slip between the number and the verdict.
- **No `dispersion` parameter.** It was considered and dropped when the
  objective became Spearman. The weighted Pearson keeps the Poisson
  delta-method weights, documented as an approximation, because it is only a
  diagnostic.
- **No reproducibility ceiling.** It needs true replicates; the F and R
  branches are different libraries, so their disagreement confounds library
  difference with measurement noise.
- **`SlurmRunner` is still a stub**, unchanged.
