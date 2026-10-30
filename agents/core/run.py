"""Run context: the unit of work, and the audit trail.

Every orchestrator invocation opens a :class:`Run`.  Everything it produces --
configs sent to Julia, results, logs, generated sequences -- lands in
``agents/store/runs/<run_id>/``, and every tool call the LLM makes is appended
to ``trace.jsonl``.

With agents that *choose* architectures and hyperparameters, the trace is not
bookkeeping: it is the only way to answer "why did this model come out like
this" a week later.  It also plugs straight into slapsci's observability
callbacks.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import workspace as ws

__all__ = ["Run", "new_run_id"]


def new_run_id(label: str = "run") -> str:
    """Timestamped, sortable, filesystem-safe."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{safe}"[:64]


class Run:
    """Context manager owning one run directory and its trace.

    >>> with Run.create("arch-sweep", goal="find a smaller model") as run:
    ...     run.trace("predict_fitness", {"n": 12}, {"mean_energy": -1.2})
    """

    def __init__(self, run_id: str, goal: str = "") -> None:
        self.run_id = ws.slug(run_id)
        self.goal = goal
        self.dir = ws.run_dir(self.run_id, create=True)
        self.started = time.time()

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, label: str = "run", goal: str = "") -> "Run":
        return cls(new_run_id(label), goal=goal)

    def __enter__(self) -> "Run":
        self._write_manifest(status="running")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.trace(
                "run.failed",
                {},
                {"error": f"{exc_type.__name__}: {exc}"},
                error="".join(traceback.format_exception(exc_type, exc, tb))[-8000:],
            )
            self._write_manifest(status="failed")
        else:
            self._write_manifest(status="completed")
        return False  # never swallow

    # -- artefacts ---------------------------------------------------------
    @property
    def artifacts(self) -> Path:
        d = self.dir / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_manifest(self, *, status: str) -> None:
        (self.dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "goal": self.goal,
                    "status": status,
                    "started": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
                    "duration_s": round(time.time() - self.started, 3),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def trace(
        self,
        tool: str,
        params: dict[str, Any],
        result: Any = None,
        *,
        error: str | None = None,
    ) -> None:
        """Append one tool call to ``trace.jsonl``.

        ``result`` is summarised, not dumped: a sampler returning 10k sequences
        should not make the trace unreadable.
        """
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "params": _summarise(params),
            "result": _summarise(result),
        }
        if error:
            record["error"] = error
        with (self.dir / "trace.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def write_summary(self, markdown: str) -> Path:
        p = self.dir / "summary.md"
        p.write_text(markdown, encoding="utf-8")
        return p


def _summarise(obj: Any, *, max_items: int = 5, max_chars: int = 600) -> Any:
    """Keep traces legible: truncate long lists and long strings."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        return {k: _summarise(v, max_items=max_items, max_chars=max_chars) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        head = [_summarise(x, max_items=max_items, max_chars=max_chars) for x in obj[:max_items]]
        if len(obj) > max_items:
            head.append(f"... (+{len(obj) - max_items} more)")
        return head
    if isinstance(obj, str) and len(obj) > max_chars:
        return obj[:max_chars] + f"... (+{len(obj) - max_chars} chars)"
    return obj
