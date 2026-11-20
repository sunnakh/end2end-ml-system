import json
from datetime import date
from pathlib import Path

import pytest

from late_delivery.config import load_settings
from late_delivery.platform.pipeline.manifest import write_manifest
from late_delivery.platform.pipeline.runner import Context, Step, new_context, run_steps

DS = date(2017, 1, 2)


def _ctx(tmp_path: Path) -> Context:
    s = load_settings("local")
    s = s.model_copy(update={"paths": s.paths.model_copy(update={"data_root": str(tmp_path)})})
    return new_context(s, DS)


def test_run_id_format(tmp_path: Path) -> None:
    rid = _ctx(tmp_path).run_id
    assert rid.startswith("20170102-") and len(rid) == 17


def test_runs_in_dependency_order(tmp_path: Path) -> None:
    order: list[str] = []
    steps = [
        Step("c", lambda c: order.append("c"), ("b",)),
        Step("a", lambda c: order.append("a")),
        Step("b", lambda c: order.append("b"), ("a",)),
    ]
    result = run_steps(steps, _ctx(tmp_path))
    assert result.ok and order == ["a", "b", "c"]
    assert result.steps_run == ["a", "b", "c"]


def test_only_runs_selected(tmp_path: Path) -> None:
    order: list[str] = []
    steps = [Step("a", lambda c: order.append("a")), Step("b", lambda c: order.append("b"), ("a",))]
    run_steps(steps, _ctx(tmp_path), only=["b"])
    assert order == ["b"]


def test_cycle_raises(tmp_path: Path) -> None:
    steps = [Step("a", lambda c: None, ("b",)), Step("b", lambda c: None, ("a",))]
    with pytest.raises(ValueError, match="cycle"):
        run_steps(steps, _ctx(tmp_path))


def test_stops_on_failure_and_reports_step(tmp_path: Path) -> None:
    order: list[str] = []

    def boom(c: Context) -> None:
        raise RuntimeError("bad data")

    steps = [Step("a", boom), Step("b", lambda c: order.append("b"), ("a",))]
    result = run_steps(steps, _ctx(tmp_path))
    assert not result.ok
    assert result.failed_step == "a"
    assert result.error is not None and "bad data" in result.error
    assert order == []


def test_manifest_has_required_keys(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = run_steps([Step("a", lambda c: None)], ctx)
    path = write_manifest(ctx, result, {"orders": "abc"})
    data = json.loads(Path(path).read_text())
    assert path == f"{tmp_path}/manifests/{ctx.run_id}.json"
    assert {
        "run_id", "ds", "steps", "started_at", "ended_at", "git_commit", "config_hash",
        "raw_hashes", "outputs", "quarantine_counts", "ok",
    } <= set(data)  # fmt: skip
    assert data["ds"] == "2017-01-02" and data["ok"] is True


def test_manifest_records_salt_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from late_delivery.platform.pii import salt_fingerprint

    monkeypatch.setenv("LATE_DELIVERY_PII_SALT", "s1")
    ctx = _ctx(tmp_path)
    path = write_manifest(ctx, run_steps([Step("a", lambda c: None)], ctx), {})
    assert json.loads(Path(path).read_text())["salt_fingerprint"] == salt_fingerprint("s1")
