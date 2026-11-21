from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from late_delivery import cli
from late_delivery.platform.storage import PartitionInfo


def test_train_wires_snapshots_experiment_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built: list[date] = []

    def fake_build(settings: Any, cutoff: date) -> PartitionInfo:
        built.append(cutoff)
        path = tmp_path / f"snap-{cutoff}"
        path.mkdir(exist_ok=True)
        pl.DataFrame({"order_id": ["a"]}).write_parquet(path / "part-0.parquet")
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
    assert seen["run_id"] == "r9"
    assert seen["models_dir"] == tmp_path / "models"
    assert seen["out_dir"] == tmp_path / "models" / "runs" / "r9"
    out = capsys.readouterr().out
    assert "BLOCKED" in out and str(tmp_path / "models" / "runs" / "r9" / "report.md") in out


def test_default_run_id_has_time_and_commit() -> None:
    run_id = cli.default_run_id("abc1234")
    assert run_id.endswith("-abc1234") and run_id[:8].isdigit()
