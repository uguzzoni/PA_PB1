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
