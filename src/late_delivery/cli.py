"""Command line entry point: backfill, rerun, single steps, training snapshots and training."""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl

from late_delivery.config import Settings, load_settings
from late_delivery.dag import DOWNSTREAM_OF_HISTORY, STEPS
from late_delivery.modeling.experiment import Snapshot, run_experiment
from late_delivery.modeling.report import write_report
from late_delivery.platform.pipeline.manifest import write_manifest
from late_delivery.platform.pipeline.runner import new_context, run_steps
from late_delivery.platform.storage import list_partitions
from late_delivery.source.simulator import RAW_FILES
from late_delivery.steps import build_training_set

log = logging.getLogger("late_delivery.cli")


@lru_cache(maxsize=4)
def _raw_hashes_cached(raw_dir: str) -> tuple[tuple[str, str], ...]:
    out = []
    for table, name in RAW_FILES.items():
        path = Path(raw_dir) / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        out.append((table, digest))
    return tuple(out)


def raw_hashes(settings: Settings) -> dict[str, str]:
    return dict(_raw_hashes_cached(str(settings.paths.raw_dir)))


def _run_day(settings: Settings, ds: date, only: list[str] | None = None) -> bool:
    ctx = new_context(settings, ds)
    result = run_steps(STEPS, ctx, only=only)
    write_manifest(ctx, result, raw_hashes(settings))
    if not result.ok:
        log.error("day %s failed at %s: %s", ds, result.failed_step, result.error)
    return result.ok


def backfill(settings: Settings, start: date, end: date) -> bool:
    """Run every step for each day in order; stop at the first failed day."""
    ds = start
    while ds <= end:
        if not _run_day(settings, ds):
            return False
        ds += timedelta(days=1)
    return True


def run_step(settings: Settings, step: str, ds: date) -> bool:
    if step not in {s.name for s in STEPS}:
        raise ValueError(f"unknown step {step!r}")
    return _run_day(settings, ds, only=[step])


def rerun(settings: Settings, ds: date) -> bool:
    """Rerun one day fully, then refresh history-dependent steps for every later processed day."""
    if not _run_day(settings, ds):
        return False
    later = [
        d for d in list_partitions(settings.paths.data_root, "silver", "orders_changelog") if d > ds
    ]
    return all(_run_day(settings, d, only=DOWNSTREAM_OF_HISTORY) for d in later)


def _date(value: str) -> date:
    return date.fromisoformat(value)


def git_commit() -> str:
    """Short commit of the checkout, marked "-dirty" if there are uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def default_run_id(commit: str) -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{commit}"


def train(settings: Settings, run_id: str | None, models_dir: Path | None) -> bool:
    """Builds the snapshots the experiment asks for, runs it and writes the report."""

    def load_snapshot(cutoff: date) -> Snapshot:
        info = build_training_set(settings, cutoff)
        frame = pl.read_parquet(f"{info.path}/part-0.parquet")
        return Snapshot(frame=frame, content_hash=info.content_hash)

    commit = git_commit()
    run_id = run_id or default_run_id(commit)
    root = models_dir or Path(settings.training.models_dir)
    result = run_experiment(
        settings.training, load_snapshot, root, run_id=run_id, git_commit=commit
    )
    out_dir = root / "runs" / run_id
    write_report(result, settings.training, out_dir)
    status = "PASSED (champion updated)" if result.gate.passed else "BLOCKED (champion kept)"
    print(f"run {run_id}: {result.chosen['model']} — gate {status}")
    print(f"report: {out_dir / 'report.md'}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="late-delivery")
    parser.add_argument("--env", default="local")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("backfill", help="run the daily pipeline for a date range")
    p.add_argument("--start", type=_date)
    p.add_argument("--end", type=_date)
    p = sub.add_parser("rerun", help="rerun one day and refresh later days")
    p.add_argument("--ds", type=_date, required=True)
    p = sub.add_parser("run-step", help="run a single step for one day")
    p.add_argument("--step", required=True)
    p.add_argument("--ds", type=_date, required=True)
    p = sub.add_parser("training-set", help="build a labeled training snapshot")
    p.add_argument("--cutoff", type=_date, required=True)
    p = sub.add_parser("train", help="train, evaluate, gate and report (M3)")
    p.add_argument("--run-id")
    p.add_argument("--models-dir", type=Path, help="defaults to training.models_dir")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings(args.env, args.config_dir)
    if args.command == "backfill":
        ok = backfill(
            settings, args.start or settings.replay.start, args.end or settings.replay.end
        )
    elif args.command == "rerun":
        ok = rerun(settings, args.ds)
    elif args.command == "run-step":
        ok = run_step(settings, args.step, args.ds)
    elif args.command == "train":
        ok = train(settings, args.run_id, args.models_dir)
    else:
        info = build_training_set(settings, args.cutoff)
        print(f"{info.path} rows={info.rows} hash={info.content_hash}")
        ok = True
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
