"""Minimal step runner: dependency-ordered steps over one partition date."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from late_delivery.config import Settings
from late_delivery.platform.logs import log_event
from late_delivery.platform.storage import PartitionInfo


@dataclass
class Context:
    settings: Settings
    ds: date
    run_id: str
    outputs: list[PartitionInfo] = field(default_factory=list)
    quarantine_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    row_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def root(self) -> str:
        return self.settings.paths.data_root


@dataclass(frozen=True)
class Step:
    name: str
    fn: Callable[[Context], None]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunResult:
    run_id: str
    ok: bool
    steps_run: list[str]
    failed_step: str | None
    error: str | None
    started_at: str
    ended_at: str


def new_context(settings: Settings, ds: date) -> Context:
    return Context(settings=settings, ds=ds, run_id=f"{ds:%Y%m%d}-{uuid.uuid4().hex[:8]}")


def topological_order(steps: list[Step]) -> list[Step]:
    by_name = {s.name: s for s in steps}
    ordered: list[Step] = []
    state: dict[str, str] = {}

    def visit(name: str) -> None:
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            raise ValueError(f"dependency cycle at step {name!r}")
        state[name] = "visiting"
        for dep in by_name[name].depends_on:
            if dep in by_name:
                visit(dep)
        state[name] = "done"
        ordered.append(by_name[name])

    for s in steps:
        visit(s.name)
    return ordered


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_steps(steps: list[Step], ctx: Context, only: list[str] | None = None) -> RunResult:
    started = _now()
    ran: list[str] = []
    for step in topological_order(steps):
        if only is not None and step.name not in only:
            continue
        t0 = time.perf_counter()
        try:
            step.fn(ctx)
        except Exception as exc:  # reported in the result and the manifest
            error = f"{type(exc).__name__}: {exc}"
            log_event(
                "step_failed",
                step=step.name,
                failure=exc,
                seconds=round(time.perf_counter() - t0, 3),
            )
            return RunResult(ctx.run_id, False, ran, step.name, error, started, _now())
        ran.append(step.name)
        log_event("step_finished", step=step.name, seconds=round(time.perf_counter() - t0, 3))
    return RunResult(ctx.run_id, True, ran, None, None, started, _now())
