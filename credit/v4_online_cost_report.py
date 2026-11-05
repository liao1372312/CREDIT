"""Derive held-out execution-call costs for the v4 main comparison table."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/credit_v3_remote_recovery_baselines_confirm_v1/raw_test_results.json"
DIRECT = ROOT / "results/credit_v4_broad_benchmark_v2/direct_test_results.json"
OUTPUT = ROOT / "results/credit_v4_online_cost_v1.json"

METHODS = (
    ("NoMem", "NoMem"),
    ("FIFO / Recency", "FIFO"),
    ("MC-Shapley", "Physical-MCShapley"),
    ("CMP-adapted", "CMP-IPW-adapted"),
    ("CAR-adapted", "CAR-Memory-adapted"),
    ("Version-filter", "Probe-Filter"),
    ("Direct-Sandbox", "Direct-Sandbox"),
    ("CREDIT", "CREDIT-Covered"),
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = json.loads(BASE.read_text()) + json.loads(DIRECT.read_text())
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    keyed: dict[str, dict[tuple[int, int, str], int]] = defaultdict(dict)
    for row in rows:
        method = row["method"]
        calls = int(row["api_calls"])
        totals[method][0] += calls
        totals[method][1] += 1
        keyed[method][row["seed"], row["cycle"], row["task"]] = calls

    if keyed["FIFO"] != keyed["Recency"]:
        raise ValueError("FIFO and Recency online calls differ")

    table = []
    for label, method in METHODS:
        calls, episodes = totals[method]
        if episodes != 24_576:
            raise ValueError(f"unexpected held-out coverage for {method}: {episodes}")
        table.append({
            "method": label,
            "raw_method": method,
            "heldout_episodes": episodes,
            "heldout_api_calls": calls,
            "online_calls_per_test": calls / episodes,
        })

    payload = {
        "metric": "physical API calls made during held-out task execution divided by held-out episodes",
        "interpretation": "descriptive load; failed workflows can terminate early",
        "excludes": "all source-history, adaptation, screening, and certification calls",
        "inputs_sha256": {str(BASE.relative_to(ROOT)): digest(BASE),
                          str(DIRECT.relative_to(ROOT)): digest(DIRECT)},
        "table": table,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
