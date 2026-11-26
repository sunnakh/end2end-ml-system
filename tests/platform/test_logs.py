import json
from datetime import datetime
from pathlib import Path

import pytest

from late_delivery.platform.logs import log_event, run_log


def test_events_flush_immediately_and_nested_runs_are_isolated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    secret = 'test-"salt'
    with run_log(str(first), component="training", run_id="one", secret=secret):
        log_event("started", model=secret)
        first_line = json.loads(first.read_text())
        assert first_line["model"] == "[redacted]"
        assert datetime.fromisoformat(first_line["ts"]).utcoffset().total_seconds() == 0
        with pytest.raises(RuntimeError), run_log(str(second), component="training", run_id="two"):
            log_event("scored", value=float("nan"))
            raise RuntimeError("simulated failure")
        log_event("finished")
    assert secret not in capsys.readouterr().err
    a, b = ([json.loads(line) for line in p.read_text().splitlines()] for p in (first, second))
    assert [e["event"] for e in a] == ["started", "finished"]
    assert [e["event"] for e in b] == ["scored", "run_failed"]
    assert all(e["run_id"] == "one" for e in a) and all(e["run_id"] == "two" for e in b)
    assert b[0]["value"] is None and b[1]["level"] == "ERROR"
