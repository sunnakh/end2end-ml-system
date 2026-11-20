"""Per-run manifest: what ran, on which inputs, producing which outputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict

import fsspec

from late_delivery.platform.pii import salt_fingerprint
from late_delivery.platform.pipeline.runner import Context, RunResult


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(ctx: Context) -> str:
    return hashlib.sha256(ctx.settings.model_dump_json().encode()).hexdigest()


def _salt_fingerprint(ctx: Context) -> str | None:
    salt = os.environ.get(ctx.settings.pii.salt_env, "")
    return salt_fingerprint(salt) if salt else None


def write_manifest(ctx: Context, result: RunResult, raw_hashes: dict[str, str]) -> str:
    path = f"{ctx.root}/manifests/{ctx.run_id}.json"
    body = {
        "run_id": ctx.run_id,
        "ds": ctx.ds.isoformat(),
        "steps": result.steps_run,
        "ok": result.ok,
        "failed_step": result.failed_step,
        "error": result.error,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "git_commit": git_commit(),
        "config_hash": config_hash(ctx),
        "salt_fingerprint": _salt_fingerprint(ctx),
        "raw_hashes": raw_hashes,
        "outputs": [asdict(o) for o in ctx.outputs],
        "quarantine_counts": ctx.quarantine_counts,
        "row_counts": ctx.row_counts,
    }
    fs, fs_path = fsspec.core.url_to_fs(path)
    fs.makedirs(fs_path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(fs_path, "w") as f:
        json.dump(body, f, indent=2, sort_keys=True)
    return path
