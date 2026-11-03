"""Agent wiring, schemas, registry and analysis -- all without an LLM."""

import pytest

from agents.core.definitions import (
    build_architect_agent,
    build_generator_agent,
    build_team,
    build_trainer_agent,
)
from agents.core.schemas import ArchSpec, Layer, SeqConstraints, SeqRecord
from agents.core.tools import analysis


class TestArchSpec:
    def _layers(self, units):
        acts = ["tanh"] * (len(units) - 1) + ["identity"]
        return [Layer(units=u, activation=a) for u, a in zip(units, acts)]

    def test_param_count(self):
        arch = ArchSpec(name="t", layers=self._layers([16, 8, 1]), seq_length=15)
        # 315*16+16 + 16*8+8 + 8*1+1 = 5056+136+9
        assert arch.n_params() == 5201
        assert arch.input_dim == 315

    def test_default_width_is_the_gapped_alphabet(self):
        """21, not 20: the campaign's models carry a trailing gap column.

        Getting this wrong makes estimate_params() disagree with the n_params
        train.jl actually reports, and makes the Architect reason about
        capacity from the wrong input dimension.
        """
        assert ArchSpec(name="t", layers=self._layers([1]), seq_length=15).n_aa == 21

    def test_no_gap_alphabet_is_still_reachable(self):
        arch = ArchSpec(name="t", layers=self._layers([16, 8, 1]), seq_length=15, n_aa=20)
        assert arch.input_dim == 300
        assert arch.n_params() == 4961

    def test_width_outside_20_or_21_is_refused(self):
        with pytest.raises(ValueError):
            ArchSpec(name="t", layers=self._layers([1]), seq_length=15, n_aa=22)

    def test_non_scalar_output_is_refused(self):
        with pytest.raises(ValueError, match="units=1"):
            ArchSpec(name="t", layers=self._layers([16, 4]), seq_length=15)

    def test_squashed_output_is_refused(self):
        with pytest.raises(ValueError, match="identity"):
            ArchSpec(name="t", layers=[Layer(units=1, activation="sigmoid")], seq_length=15)

    def test_sequence_validation_rejects_nonstandard_residues(self):
        with pytest.raises(ValueError, match="non-standard"):
            SeqRecord(seq="ACDEFX", energy=0.0)


class TestAgentDefinitions:
    @pytest.mark.parametrize(
        "build", [build_architect_agent, build_trainer_agent, build_generator_agent]
    )
    def test_tools_expose_valid_json_schemas(self, build):
        agent = build()
        assert agent.tools, f"{agent.name} has no tools"
        for t in agent.tools:
            schema = t.input_schema
            assert schema["type"] == "object"
            assert t.description.strip(), f"{t.name} has no description for the LLM"

    def test_unknown_tool_lists_what_is_available(self):
        agent = build_generator_agent()
        with pytest.raises(KeyError, match="has no tool"):
            agent.call_tool("definitely_not_a_tool")

    def test_openai_and_anthropic_shapes(self):
        agent = build_generator_agent()
        oai = agent.to_openai_tools()
        ant = agent.to_anthropic_tools()
        assert all(t["type"] == "function" for t in oai)
        assert all("input_schema" in t for t in ant)

    def test_generator_has_the_phase1_tools(self):
        names = set(build_generator_agent().tool_map())
        assert {"predict_fitness", "generate_sequences", "check_diversity"} <= names

    def test_tool_parameter_named_name_does_not_collide(self):
        """Regression: save_candidates takes `name`, and so did call_tool."""
        agent = build_generator_agent()
        with pytest.raises(ValueError, match="energies"):
            agent.call_tool(
                "save_candidates",
                sequences=["ADEFGHIKLMNPQRS"],
                energies=[],
                name="collision-check",
            )

    def test_bad_arguments_are_rejected_before_the_call(self):
        agent = build_generator_agent()
        with pytest.raises(Exception):
            agent.call_tool("check_diversity", sequences="not-a-list", identity_threshold="high")


class TestAnalysis:
    def test_collapse_is_visible(self):
        collapsed = ["ACDEFGHIKLMNPQR"] * 10
        m = analysis.diversity_metrics(collapsed)
        assert m.n_unique == 1
        assert m.min_hamming == 0
        assert m.n_clusters == 1
        assert m.mean_position_entropy == 0.0

    def test_diverse_set_scores_higher(self):
        import random

        rng = random.Random(0)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        diverse = ["".join(rng.choice(alphabet) for _ in range(15)) for _ in range(20)]
        m = analysis.diversity_metrics(diverse)
        assert m.n_unique == 20
        assert m.mean_hamming > 10
        assert m.mean_position_entropy > 2.0

    def test_empty_input(self):
        assert analysis.diversity_metrics([]).n == 0

    def test_hamming_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            analysis.hamming("AC", "ACD")

    def test_filter_drops_forbidden_and_duplicates(self):
        recs = [
            SeqRecord(seq="ADEFGHIKLMNPQRS", energy=1.0),
            SeqRecord(seq="ADEFGHIKLMNPQRS", energy=1.0),   # duplicate
            SeqRecord(seq="CDEFGHIKLMNPQRS", energy=0.5),   # contains C
        ]
        kept = analysis.filter_sequences(recs, SeqConstraints(forbidden_aa="C"))
        assert len(kept) == 1

    def test_ranking_puts_lowest_energy_first(self):
        recs = [SeqRecord(seq="A" * 15, energy=e) for e in (3.0, 1.0, 2.0)]
        assert [r.energy for r in analysis.rank_sequences(recs)] == [1.0, 2.0, 3.0]


class TestTeam:
    def test_team_describes_its_agents(self, tmp_path, monkeypatch):
        from agents.core import workspace as ws

        monkeypatch.setattr(ws, "STORE", tmp_path / "store")
        monkeypatch.setattr(ws, "RUNS_DIR", tmp_path / "store" / "runs")
        team = build_team(goal="smoke", label="test")
        desc = team.describe()
        assert set(desc) == {"architect", "trainer", "generator"}
        assert desc["generator"]["tools"]
