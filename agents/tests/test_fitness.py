"""Encoding, weight loading and the sampler.

The orientation test matters most: Julia stores Dense(in, out) as (out, in)
and is column-major, so an exported matrix can arrive either way round. The
loader must handle both and must refuse anything that chains with neither.
"""

import json

import numpy as np
import pytest

from agents.core.schemas import AA_ALPHABET, SampleConfig, SeqConstraints
from agents.core.tools import fitness


def make_weights(path, dims=(300, 16, 8, 1), transpose=False):
    layers = []
    rng = np.random.default_rng(0)
    for a, b in zip(dims, dims[1:]):
        w = rng.normal(size=(a, b)) * 0.1
        layers.append({"W": (w.T if transpose else w).tolist(), "b": rng.normal(size=b).tolist()})
    path.write_text(json.dumps({"layers": layers}), encoding="utf-8")
    return path


class TestEncoding:
    def test_shape_and_one_hot(self):
        x = fitness.encode_onehot(["ACDEFGHIKLMNPQR"], 15)
        assert x.shape == (1, 300)
        assert x.sum() == 15
        assert set(np.unique(x)) <= {0.0, 1.0}

    def test_first_residue_lands_at_its_alphabet_index(self):
        x = fitness.encode_onehot(["A" * 15], 15).reshape(15, 20)
        assert x[0, AA_ALPHABET.index("A")] == 1.0

    def test_rejects_unknown_residue(self):
        with pytest.raises(ValueError, match="unknown residue"):
            fitness.encode_onehot(["X" * 15], 15)

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="length"):
            fitness.encode_onehot(["ACDEF"], 15)


class TestLoader:
    @pytest.mark.parametrize("transpose", [False, True])
    def test_both_orientations_load(self, tmp_path, transpose):
        p = make_weights(tmp_path / "w.json", transpose=transpose)
        model = fitness.load_energy_model(p, seq_length=15)
        e = model.energy(fitness.encode_onehot(["ACDEFGHIKLMNPQR"], 15))
        assert e.shape == (1,)
        assert np.isfinite(e).all()

    def test_mismatched_dimensions_are_refused(self, tmp_path):
        p = make_weights(tmp_path / "w.json", dims=(123, 16, 1))
        with pytest.raises(ValueError, match="chains with neither orientation"):
            fitness.load_energy_model(p, seq_length=15)

    def test_unknown_layout_names_the_keys(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"mystery": [1, 2, 3]}), encoding="utf-8")
        with pytest.raises(ValueError, match="unrecognised weights layout"):
            fitness.load_energy_model(p, seq_length=15)

    def test_inspect_reports_structure(self, tmp_path):
        p = make_weights(tmp_path / "w.json")
        info = fitness.inspect_weights_file(p)
        assert info["top_level_keys"] == ["layers"]

    def test_batch_scoring_is_consistent(self, tmp_path):
        p = make_weights(tmp_path / "w.json")
        model = fitness.load_energy_model(p, seq_length=15)
        seqs = ["ACDEFGHIKLMNPQR", "WWWWWWWWWWWWWWW"]
        together = model.energy(fitness.encode_onehot(seqs, 15))
        apart = [model.energy(fitness.encode_onehot([s], 15))[0] for s in seqs]
        assert np.allclose(together, apart)


class TestExportedLayout:
    """The layout agents/julia/train.jl writes and model_3layer.py reads.

    Weights are stored as (in, out) -- Julia serialises a Flux (out, in) matrix
    column-major -- and the campaign's models are 21 residues wide, with the
    gap column last and never occupied. Getting either wrong produces confident
    nonsense rather than an error, which is why these are pinned.
    """

    def write_dense(self, path, dims=(315, 45, 15, 1), arch=None):
        rng = np.random.default_rng(1)
        payload = {}
        for k, (a, b) in enumerate(zip(dims, dims[1:]), start=1):
            payload[f"dense{k}_W"] = (rng.normal(size=(a, b)) * 0.1).tolist()
            payload[f"dense{k}_b"] = rng.normal(size=b).tolist()
        if arch is not None:
            payload["arch"] = arch
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_dense_layout_loads_as_21_wide(self, tmp_path):
        model = fitness.load_energy_model(self.write_dense(tmp_path / "w.json"), seq_length=15)
        assert model.n_aa == 21
        assert model.input_dim == 315
        e = model.energy(fitness.encode_onehot(["ACDEFGHIKLMNPQR"], 15, model.n_aa))
        assert e.shape == (1,) and np.isfinite(e).all()

    def test_arch_block_supplies_activations_and_width(self, tmp_path):
        p = self.write_dense(
            tmp_path / "w.json",
            arch={"seq_length": 15, "n_aa": 21, "units": [45, 15, 1],
                  "activations": ["tanh", "tanh", "identity"]},
        )
        model = fitness.load_energy_model(p)
        assert model.activations == ["tanh", "tanh", "identity"]
        assert model.n_aa == 21 and model.seq_length == 15

    def test_activations_default_to_the_campaign_convention(self, tmp_path):
        model = fitness.load_energy_model(self.write_dense(tmp_path / "w.json"), seq_length=15)
        assert model.activations == ["relu", "relu", "identity"]

    def test_gap_column_is_encoded_but_never_set(self):
        x = fitness.encode_onehot(["A" * 15], 15, 21).reshape(15, 21)
        assert x.shape == (15, 21)
        assert x[:, 20].sum() == 0.0, "the gap column must stay empty for a designed binder"
        assert x[0, AA_ALPHABET.index("A")] == 1.0

    def test_encode_rejects_other_widths(self):
        with pytest.raises(ValueError, match="n_aa must be"):
            fitness.encode_onehot(["A" * 15], 15, 22)

    def test_promoted_model_matches_colabdesign(self):
        """The agent fast path and the ColabDesign energy term must agree.

        Both read the same file and implement the same forward pass, so any
        disagreement is a real bug. The expected value was cross-checked
        against colabdesign/energy_model/model_3layer.py and against the Flux
        chain rebuilt by agents/julia/common.jl.
        """
        from agents.core import workspace as ws

        p = ws.REPO_ROOT / "data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json"
        if not p.exists():
            pytest.skip("promoted weights not present in this checkout")
        model = fitness.load_energy_model(p, seq_length=15)
        assert model.n_aa == 21
        e = model.energy(fitness.encode_onehot(["ACDEFGHIKLMNPQR"], 15, model.n_aa))[0]
        assert e == pytest.approx(9.080465, abs=1e-4)


class TestSampler:
    def _model(self, tmp_path):
        return fitness.load_energy_model(make_weights(tmp_path / "w.json"), seq_length=15)

    def test_returns_requested_count_and_length(self, tmp_path):
        cfg = SampleConfig(model_name="demo", n=8, length=15, n_steps=50, seed=1)
        res = fitness.sample_mcmc_python(self._model(tmp_path), cfg)
        assert len(res.sequences) == 8
        assert all(len(r.seq) == 15 for r in res.sequences)
        assert 0.0 <= res.accepted_fraction <= 1.0

    def test_respects_forbidden_residues(self, tmp_path):
        cfg = SampleConfig(
            model_name="demo", n=10, length=15, n_steps=100, seed=2,
            constraints=SeqConstraints(forbidden_aa="CWY"),
        )
        res = fitness.sample_mcmc_python(self._model(tmp_path), cfg)
        assert all(not (set("CWY") & set(r.seq)) for r in res.sequences)

    def test_respects_fixed_positions(self, tmp_path):
        cfg = SampleConfig(
            model_name="demo", n=6, length=15, n_steps=80, seed=3,
            constraints=SeqConstraints(forbidden_aa="C", fixed_positions={0: "M", 14: "K"}),
        )
        res = fitness.sample_mcmc_python(self._model(tmp_path), cfg)
        assert all(r.seq[0] == "M" and r.seq[14] == "K" for r in res.sequences)

    def test_minimisation_improves_on_random(self, tmp_path):
        """The sampler must actually optimise, not just wander."""
        model = self._model(tmp_path)
        cfg = SampleConfig(model_name="demo", n=16, length=15, n_steps=400, seed=4,
                           temperature=0.5)
        res = fitness.sample_mcmc_python(model, cfg)
        sampled = np.mean([r.energy for r in res.sequences])

        rng = np.random.default_rng(99)
        random_seqs = ["".join(rng.choice(list(AA_ALPHABET), 15)) for _ in range(16)]
        baseline = np.mean(model.energy(fitness.encode_onehot(random_seqs, 15)))

        assert sampled < baseline, "MCMC did not improve on random sequences"

    def test_seeds_are_honoured(self, tmp_path):
        cfg = SampleConfig(model_name="demo", n=2, length=15, n_steps=1, seed=5,
                           temperature=1e-9, seeds=["ACDEFGHIKLMNPQR"])
        res = fitness.sample_mcmc_python(self._model(tmp_path), cfg)
        assert len(res.sequences) == 2

    def test_reproducible_with_same_seed(self, tmp_path):
        model = self._model(tmp_path)
        cfg = SampleConfig(model_name="demo", n=4, length=15, n_steps=60, seed=7)
        a = fitness.sample_mcmc_python(model, cfg)
        b = fitness.sample_mcmc_python(model, cfg)
        assert [r.seq for r in a.sequences] == [r.seq for r in b.sequences]
