"""Configuration loading.

The point of these is the degradation path: a missing, partial or broken
config file must leave the system working with the values it had before,
because a typo in YAML should not take down a training run.
"""

import pytest
import yaml

from agents.core import config


class TestLoading:
    def test_shipped_file_is_what_load_config_reads(self):
        assert config.config_path().name == "default.yaml"
        assert config.config_path().is_file()

    def test_shipped_file_introduces_no_unknown_key(self):
        """DEFAULTS must describe every key the YAML sets.

        Otherwise the fallback is not a complete configuration and the
        degradation path quietly stops working.
        """
        shipped = yaml.safe_load(config.config_path().read_text(encoding="utf-8"))

        def walk(got, known, path=""):
            for key, value in got.items():
                where = f"{path}.{key}" if path else key
                assert key in known, f"{where} is in default.yaml but not in DEFAULTS"
                if isinstance(value, dict):
                    walk(value, known[key], where)

        walk(shipped, config.DEFAULTS)

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        merged = config.load_config(tmp_path / "does-not-exist.yaml")
        assert merged == config.DEFAULTS

    def test_unparseable_file_warns_and_falls_back(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("llm: [unclosed\n", encoding="utf-8")
        with pytest.warns(RuntimeWarning, match="could not parse"):
            merged = config.load_config(bad)
        assert merged == config.DEFAULTS

    def test_partial_file_merges_rather_than_replaces(self, tmp_path):
        p = tmp_path / "partial.yaml"
        p.write_text("sampling:\n  n: 7\n", encoding="utf-8")
        merged = config.load_config(p)
        assert merged["sampling"]["n"] == 7
        # the siblings survive
        assert merged["sampling"]["n_steps"] == config.DEFAULTS["sampling"]["n_steps"]
        assert merged["execution"]["backend"] == config.DEFAULTS["execution"]["backend"]

    def test_env_var_overrides_the_path(self, tmp_path, monkeypatch):
        p = tmp_path / "other.yaml"
        p.write_text("execution:\n  julia_bin: julia-1.11\n", encoding="utf-8")
        monkeypatch.setenv(config.CONFIG_ENV_VAR, str(p))
        assert config.config_path() == p
        assert config.load_config(reload=True)["execution"]["julia_bin"] == "julia-1.11"
        monkeypatch.delenv(config.CONFIG_ENV_VAR)
        config.load_config(reload=True)   # restore the module cache for other tests


class TestGet:
    def test_dotted_lookup(self):
        assert config.get("sampling.n_steps") == config.DEFAULTS["sampling"]["n_steps"]
        assert config.get("execution.backend") == "local"

    def test_missing_key_returns_the_fallback(self):
        assert config.get("nope.not.here", "fallback") == "fallback"
        assert config.get("sampling.not_a_key") is None

    def test_a_null_in_yaml_is_a_value_not_a_miss(self):
        assert config.get("llm.per_agent.architect", "unused") is None


class TestConsumers:
    def test_runner_defaults_come_from_config(self):
        from agents.core.execution import LocalJuliaRunner

        r = LocalJuliaRunner()
        assert r.julia_bin == config.get("execution.julia_bin")
        assert r.timeout_s == config.get("execution.timeout_s")
        assert r.project.name == config.get("execution.julia_project")

    def test_unknown_backend_is_refused(self, monkeypatch):
        from agents.core import execution

        monkeypatch.setattr(execution.cfg, "get",
                            lambda key, default=None: "nonsense"
                            if key == "execution.backend" else default)
        with pytest.raises(execution.ExecutionError, match="unknown execution.backend"):
            execution.default_runner()

    def test_slurm_backend_says_it_is_not_implemented(self, monkeypatch):
        from agents.core import execution

        monkeypatch.setattr(execution.cfg, "get",
                            lambda key, default=None: "slurm"
                            if key == "execution.backend" else default)
        with pytest.raises(execution.ExecutionError, match="phase-2 placeholder"):
            execution.default_runner()

    def test_sampling_defaults_reach_the_tool_schema(self):
        from agents.core.definitions import build_generator_agent

        schema = build_generator_agent().tool_map()["generate_sequences"].input_schema
        assert schema["properties"]["n_steps"]["default"] == config.get("sampling.n_steps")
        assert schema["properties"]["forbidden_aa"]["default"] == config.get("defaults.forbidden_aa")

    def test_agents_take_their_model_from_config(self):
        from agents.core.definitions import build_architect_agent

        assert build_architect_agent().model == config.get("llm.model")
        assert build_architect_agent("explicit-model").model == "explicit-model"
