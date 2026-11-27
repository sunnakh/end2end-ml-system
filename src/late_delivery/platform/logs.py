"""Structured, flushed events scoped to one run, including nested run contexts."""

from __future__ import annotations

import json
import logging
import math
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime

import fsspec


@dataclass(frozen=True)
class _RunLog:
    logger: logging.Logger
    context: dict[str, object]
    secret: str


_active: ContextVar[_RunLog | None] = ContextVar("run_logger", default=None)


def _safe_field(value: object, secret: str) -> object:
    if isinstance(value, str) and secret:
        return value.replace(secret, "[redacted]")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _safe_field(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_field(v, secret) for v in value]
    return value


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="seconds")
        return json.dumps(
            {"ts": ts.replace("+00:00", "Z"), "level": record.levelname, **vars(record)["payload"]},
            allow_nan=False,
        )


def _error_fields(exc: BaseException) -> dict[str, str]:
    """Keep all frame locations and chained errors. Library errors may embed input rows."""
    parts = []
    seen = set()
    current: BaseException | None = exc
    error = ""
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frames = list(traceback.walk_tb(current.__traceback__))
        origin = frames[-1][0].f_globals.get("__name__", "") if frames else ""
        error_module = type(current).__module__
        # Our own diagnostics describe schema, configuration and file locations.
        # A parser or validator may instead include the raw offending values.
        if (
            origin.startswith("late_delivery.")
            and (error_module == "builtins" or error_module.startswith("late_delivery."))
            and current.__cause__ is None
        ):
            message = str(current)
        elif isinstance(current, OSError) and current.strerror:
            message = current.strerror
        else:
            message = "Exception detail omitted to protect input data"
        if current is exc:
            error = message
        lines = ["Traceback (most recent call last):\n"]
        for frame, line in frames:
            lines.append(
                f'  File "{frame.f_code.co_filename}", line {line}, in {frame.f_code.co_name}\n'
            )
        lines.append(f"{type(current).__name__}: {message}")
        parts.append("".join(lines))
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return {
        "error_type": type(exc).__name__,
        "error": error,
        "traceback": "\n\nCaused by:\n\n".join(parts),
    }


def log_event(event: str, *, failure: BaseException | None = None, **fields: object) -> None:
    run = _active.get()
    if run is None:
        return
    if failure is not None:
        fields.update(_error_fields(failure))
    payload = {**run.context, "event": event, **fields}
    clean = {k: _safe_field(v, run.secret) for k, v in payload.items()}
    summary = " ".join(
        f"{k}={clean[k]}"
        for k in ("ds", "step", "model", "passed", "error_type", "seconds")
        if k in clean
    )
    run.logger.log(
        logging.ERROR if failure is not None else logging.INFO,
        f"{clean['component']} {event} run_id={clean['run_id']} {summary}".rstrip(),
        extra={"payload": clean},
    )


@contextmanager
def run_log(
    path: str, *, component: str, run_id: str, secret: str = "", **context: object
) -> Iterator[None]:
    # Each run owns its handlers. Overlapping runs never receive each other's events.
    logger = logging.Logger(component, level=logging.INFO)
    with fsspec.open(path, "at", encoding="utf-8", auto_mkdir=True) as stream:
        file_handler = logging.StreamHandler(stream)
        file_handler.setFormatter(_JSONFormatter())
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())
        token = _active.set(
            _RunLog(logger, {"component": component, "run_id": run_id, **context}, secret)
        )
        try:
            yield
        except Exception as exc:
            log_event("run_failed", failure=exc)
            raise
        finally:
            _active.reset(token)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
