"""Prove the external-effect crash window and an idempotent mitigation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_scenario(root: Path, mode: str, worker: Path) -> dict[str, object]:
    scenario = root / mode
    scenario.mkdir(parents=True, exist_ok=True)
    database = scenario / "dbos.sqlite"
    ledger = scenario / "remote-ledger.log"
    marker = scenario / "crashed-after-commit.marker"
    workflow_id = f"project15-crash-{mode}"
    command = [sys.executable, str(worker), mode, str(database), str(ledger), str(marker), workflow_id]

    first = subprocess.run(command, text=True, capture_output=True, check=False)
    if first.returncode != 71:
        raise AssertionError(f"first {mode} worker should crash with 71: {first.returncode}\n{first.stderr}")
    after_crash = ledger.read_text().splitlines()

    replacement = subprocess.run(command, text=True, capture_output=True, check=False)
    if replacement.returncode != 0:
        raise AssertionError(f"replacement {mode} worker failed: {replacement.returncode}\n{replacement.stderr}")
    recovered = json.loads(replacement.stdout)
    after_recovery = ledger.read_text().splitlines()
    expected_count = 1 if mode == "idempotent" else 2
    assert len(after_crash) == 1
    assert len(after_recovery) == expected_count
    assert recovered["result"] == expected_count
    result = {
        "mode": mode,
        "first_returncode": first.returncode,
        "after_crash": after_crash,
        "replacement_returncode": replacement.returncode,
        "after_recovery": after_recovery,
        "recovered": recovered,
    }
    (scenario / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (scenario / "first.stderr.txt").write_text(first.stderr)
    (scenario / "replacement.stderr.txt").write_text(replacement.stderr)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("crash_effect_worker.py")
    results = [run_scenario(args.evidence_dir, mode, worker) for mode in ("unprotected", "idempotent")]
    summary = {
        "unprotected_attempts": len(results[0]["after_recovery"]),
        "idempotent_commits": len(results[1]["after_recovery"]),
        "conclusion": "DBOS retries the ambiguous step; remote idempotency prevents duplicate commit",
    }
    (args.evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
