"""Execution boundary: how Python hands work to Julia (and later, to SLURM).

The contract is deliberately process-based rather than in-process
(``juliacall``):

    config.json  ->  <some process>  ->  result.json

Because the boundary is a pair of JSON files, swapping :class:`LocalJuliaRunner`
for :class:`SlurmRunner` changes *nothing* above this module.  That is the whole
reason to pay the Julia start-up cost instead of embedding the runtime.

If per-call latency ever becomes the bottleneck, the mitigations in order of
effort are: a ``PackageCompiler.jl`` sysimage, a persistent Julia daemon, and
only then ``juliacall``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from . import workspace as ws

__all__ = ["Runner", "LocalJuliaRunner", "SlurmRunner", "ExecutionError", "default_runner"]

T = TypeVar("T", bound=BaseModel)


class ExecutionError(RuntimeError):
    """A backend process failed, timed out, or returned unusable output."""


class Runner(ABC):
    """Executes one ``agents/julia/<script>`` against a config, returns a model.

    Implementations must be side-effect free outside ``run_dir``.
    """

    @abstractmethod
    def run(
        self,
        script: str,
        config: BaseModel,
        out_model: type[T],
        *,
        run_dir: Path,
        tag: str = "",
    ) -> T:
        ...


def _prepare_io(config: BaseModel, run_dir: Path, script: str, tag: str) -> tuple[Path, Path]:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(script).stem}{('_' + tag) if tag else ''}_{int(time.time() * 1000)}"
    cfg_path = artifacts / f"{stem}.config.json"
    out_path = artifacts / f"{stem}.result.json"
    cfg_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return cfg_path, out_path


def _parse_result(out_path: Path, out_model: type[T], stdout: str, stderr: str) -> T:
    if not out_path.exists():
        raise ExecutionError(
            f"backend produced no result file at {out_path}\n"
            f"--- stdout ---\n{stdout[-4000:]}\n--- stderr ---\n{stderr[-4000:]}"
        )
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExecutionError(f"result file is not valid JSON: {out_path}: {exc}") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise ExecutionError(f"backend reported an error: {payload['error']}")
    return out_model.model_validate(payload)


class LocalJuliaRunner(Runner):
    """Runs ``julia --project=<project> agents/julia/<script> --config ... --out ...``.

    Parameters
    ----------
    project:
        Julia project to activate. Defaults to the ``PD_energy_model`` package
        already in the repo, so the agent layer adds no Julia dependencies of
        its own and leaves that nested git repo untouched.
    """

    def __init__(
        self,
        project: Path | str | None = None,
        julia_bin: str = "julia",
        timeout_s: int = 3600,
        env: dict[str, str] | None = None,
    ) -> None:
        self.project = Path(project) if project else ws.read_path("PD_energy_model")
        self.julia_bin = julia_bin
        self.timeout_s = timeout_s
        self.env = env

    def available(self) -> bool:
        """True if the Julia binary can be found. Tools use this to degrade gracefully."""
        return shutil.which(self.julia_bin) is not None

    def run(
        self,
        script: str,
        config: BaseModel,
        out_model: type[T],
        *,
        run_dir: Path,
        tag: str = "",
    ) -> T:
        script_path = ws.JULIA_DIR / script
        if not script_path.exists():
            raise ExecutionError(f"no such Julia entry point: {script_path}")
        if not self.available():
            raise ExecutionError(
                f"julia binary {self.julia_bin!r} not found on PATH. "
                "Install Julia, or use the pure-Python fast path where one exists "
                "(see agents/core/tools/fitness.py)."
            )

        cfg_path, out_path = _prepare_io(config, run_dir, script, tag)
        cmd = [
            self.julia_bin,
            f"--project={self.project}",
            str(script_path),
            "--config",
            str(cfg_path),
            "--out",
            str(out_path),
        ]

        env = {**os.environ, **(self.env or {})}
        try:
            proc = subprocess.run(
                cmd,
                cwd=run_dir,          # the process can only sensibly write here
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                f"{script} exceeded {self.timeout_s}s. Reduce the workload or move it to the cluster."
            ) from exc

        (run_dir / "artifacts" / f"{cfg_path.stem}.log").write_text(
            f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
            encoding="utf-8",
        )

        if proc.returncode != 0:
            raise ExecutionError(
                f"{script} exited with code {proc.returncode}\n"
                f"--- stderr ---\n{proc.stderr[-4000:]}"
            )
        return _parse_result(out_path, out_model, proc.stdout, proc.stderr)


class SlurmRunner(Runner):
    """Phase 2: the same contract, submitted to the cluster.

    Deliberately left unimplemented. The point of this class existing now is
    that the signature is already fixed: when ColabDesign work moves to the
    cluster, nothing above :class:`Runner` needs to change.
    """

    def __init__(self, partition: str = "gpu", poll_s: int = 30, **kwargs) -> None:
        self.partition = partition
        self.poll_s = poll_s
        self.kwargs = kwargs

    def run(self, script, config, out_model, *, run_dir, tag=""):  # noqa: D102
        raise NotImplementedError(
            "SlurmRunner is a phase-2 placeholder. Implement as: write config.json, "
            "sbatch a wrapper that runs the same entry point, poll sacct until the job "
            "leaves the queue, then parse result.json exactly as LocalJuliaRunner does."
        )


def default_runner() -> Runner:
    """The runner used when a tool is not given one explicitly."""
    return LocalJuliaRunner()
