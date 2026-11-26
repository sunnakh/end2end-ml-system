import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from late_delivery import cli
from late_delivery.config import load_settings
from late_delivery.platform.storage import PartitionInfo
from tests.modeling.conftest import synthetic_snapshot


def test_train_wires_snapshots_experiment_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[date] = []

    def fake_build(settings: Any, cutoff: date) -> PartitionInfo:
        built.append(cutoff)
        path = tmp_path / f"snap-{cutoff}"
        path.mkdir(exist_ok=True)
        pl.DataFrame({"order_id": ["a"]}).write_parquet(path / "part-0.parquet")
        (path / "_meta.json").write_text('{"excluded_not_yet_due": 7}')
        return PartitionInfo(path=str(path), rows=1, content_hash=f"h{cutoff}")

    seen: dict[str, Any] = {}

    def fake_run(cfg: Any, load: Any, models_dir: Path, *, run_id: str, git_commit: str,
                 grid: Any = None) -> Any:  # fmt: skip
        snap = load(date(2018, 6, 1))
        seen.update(models_dir=models_dir, run_id=run_id, commit=git_commit, snap=snap)
        return SimpleNamespace(gate=SimpleNamespace(passed=False), chosen={"model": "m"})

    def fake_report(result: Any, cfg: Any, out_dir: Path) -> None:
        seen["out_dir"] = out_dir

    monkeypatch.setattr(cli, "build_training_set", fake_build)
    monkeypatch.setattr(cli, "run_experiment", fake_run)
    monkeypatch.setattr(cli, "write_report", fake_report)
    code = cli.main(["train", "--run-id", "r9", "--models-dir", str(tmp_path / "models")])

    assert code == 0
    assert built == [date(2018, 6, 1)]
    assert seen["snap"].content_hash == "h2018-06-01"
    assert seen["snap"].frame["order_id"].to_list() == ["a"]
    assert seen["snap"].meta == {"excluded_not_yet_due": 7}
    assert seen["run_id"] == "r9"
    assert seen["models_dir"] == tmp_path / "models"
    assert seen["out_dir"] == tmp_path / "models" / "runs" / "r9"
    out = capsys.readouterr().out
    assert "BLOCKED" in out and str(tmp_path / "models" / "runs" / "r9" / "report.md") in out


def test_default_run_id_has_time_and_commit() -> None:
    run_id = cli.default_run_id("abc1234")
    assert run_id.endswith("-abc1234") and run_id[:8].isdigit()


def test_training_log_matches_real_experiment_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = synthetic_snapshot(9000, seed=11)
    settings = load_settings("local")
    settings = settings.model_copy(
        update={
            "training": settings.training.model_copy(
                update={"window_months": [9], "bootstrap_resamples": 10}
            )
        }
    )

    def build(_settings: Any, cutoff: date) -> PartitionInfo:
        part = frame.filter(pl.col("handoff_date") < cutoff).drop("handoff_date")
        path = tmp_path / f"snap-{cutoff}"
        path.mkdir(exist_ok=True)
        part.write_parquet(path / "part-0.parquet")
        return PartitionInfo(path=str(path), rows=part.height, content_hash=f"hash-{cutoff}")

    monkeypatch.setattr(cli, "build_training_set", build)
    for run_id, expected in [("first", "champion_written"), ("repeat", "champion_kept")]:
        assert cli.train(settings, run_id, tmp_path)
        folder = tmp_path / "runs" / run_id
        metrics = json.loads((folder / "metrics.json").read_text())
        events = [
            json.loads(line) for line in (folder / "train.log.jsonl").read_text().splitlines()
        ]
        assert all(e["run_id"] == run_id and e["component"] == "training" for e in events)
        assert [e["event"] for e in events][-3:] == [expected, "report_written", "run_finished"]
        snapshots = [e for e in events if e["event"] == "snapshot_loaded"]
        assert len(snapshots) == 4
        assert all(e["content_hash"] == f"hash-{e['cutoff']}" and e["rows"] > 0 for e in snapshots)
        candidates = [e for e in events if e["event"] == "candidate_scored"]

        def scores(rows: list[dict]) -> list[tuple]:
            return sorted(
                (r["model"], r["window_months"], r["recall_at_k"], r["pr_auc"]) for r in rows
            )

        assert scores(candidates) == scores(metrics["candidates"])
        for event, field in [("gate_checked", "gate"), ("calendar_check", "calendar_gate")]:
            (check,) = [e for e in events if e["event"] == event]
            assert {k: check[k] for k in ("passed", "checks")} == metrics[field]


def test_failed_training_leaves_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = "private-training-salt"
    settings = load_settings("local")
    monkeypatch.setenv(settings.pii.salt_env, token)

    def malformed(*args: Any) -> None:
        raise pl.exceptions.ComputeError(f"bad row: customer-123, {token}")

    monkeypatch.setattr(cli, "build_training_set", malformed)
    with pytest.raises(pl.exceptions.ComputeError):
        cli.train(settings, "broken", tmp_path)
    folder = tmp_path / "runs" / "broken"
    assert not (folder / "run.json").exists()
    contents = (folder / "train.log.jsonl").read_text()
    assert token not in contents and "customer-123" not in contents
    events = [json.loads(line) for line in contents.splitlines()]
    assert [e["event"] for e in events] == ["run_started", "run_failed"]
    assert events[-1]["error_type"] == "ComputeError" and "in malformed" in events[-1]["traceback"]
