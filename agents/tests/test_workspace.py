"""The compartmentalisation guarantee, tested.

If any of these fail, the sandbox is not a sandbox.
"""

import pytest

from agents.core import workspace as ws


class TestSlug:
    @pytest.mark.parametrize("name", ["pnb-2r-3lay", "model_1", "a", "PNB.v2", "x" * 64])
    def test_accepts_safe_identifiers(self, name):
        assert ws.slug(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "../etc/passwd",
            "..",
            ".",
            "",
            "a/b",
            "a\\b",
            ".hidden",
            "x" * 65,
            "model name",
            "model;rm -rf /",
            "-leading-dash",
        ],
    )
    def test_rejects_unsafe_identifiers(self, name):
        with pytest.raises(ws.InvalidName):
            ws.slug(name)

    def test_rejects_non_string(self):
        with pytest.raises(ws.InvalidName):
            ws.slug(None)  # type: ignore[arg-type]


class TestWritePath:
    def test_stays_in_store(self):
        p = ws.write_path("models", "demo", "weights.json")
        assert p.is_relative_to(ws.STORE.resolve())

    @pytest.mark.parametrize(
        "parts",
        [
            ("..", "..", "etc", "passwd"),
            ("../../../data/energy_model_params",),
            ("models", "..", "..", "..", "src"),
        ],
    )
    def test_traversal_is_refused(self, parts):
        with pytest.raises(ws.PathEscape):
            ws.write_path(*parts)

    def test_absolute_path_cannot_escape(self):
        # joinpath with an absolute part replaces the base -- the guard catches it.
        with pytest.raises(ws.PathEscape):
            ws.write_path("/etc/passwd")

    def test_symlink_out_of_store_is_refused(self, tmp_path, monkeypatch):
        store = tmp_path / "store"
        store.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (store / "escape").symlink_to(outside)

        monkeypatch.setattr(ws, "STORE", store)
        with pytest.raises(ws.PathEscape):
            ws.write_path("escape", "secret.txt")


class TestReadPath:
    def test_allows_repo_files(self):
        p = ws.read_path("data", "input_sequences.txt")
        assert p.is_relative_to(ws.REPO_ROOT.resolve())

    def test_refuses_outside_repo(self):
        with pytest.raises(ws.PathEscape):
            ws.read_path("..", "..", "..", "etc", "passwd")


def test_store_is_inside_agents_root():
    assert ws.STORE.is_relative_to(ws.AGENTS_ROOT)
    assert ws.AGENTS_ROOT.parent == ws.REPO_ROOT


def test_no_generic_filesystem_tool_is_exposed():
    """The primary mechanism: agents have no tool that takes a raw path.

    If this fails someone added a tool that hands path control back to the
    LLM -- which defeats workspace.py entirely.
    """
    from agents.core.definitions import (
        build_architect_agent,
        build_generator_agent,
        build_trainer_agent,
    )

    banned = {"read_file", "write_file", "bash", "shell", "run_command", "glob", "listdir"}
    for build in (build_architect_agent, build_trainer_agent, build_generator_agent):
        names = set(build().tool_map())
        assert not (names & banned), f"generic filesystem tool exposed: {names & banned}"
