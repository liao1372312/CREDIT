"""Disclosed-task development check for argument-aware fast contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from . import toolsandbox_local_v3 as adapter
from . import toolsandbox_m_contract_fast_v2_confirmation as suite
from . import toolsandbox_m_contract_fast_v3 as v3
from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_m_followup import initial_job_v2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--seed", type=int, default=2701)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    source = Path("results/toolsandbox_m_contract_fast_v2_confirmation_v1")
    protocol = json.loads((source / "frozen_protocol.json").read_text())
    u = adapter.upstream_imports(Path(protocol["upstream_directory"]))
    adapter.install_local_clock(protocol["clock"])
    scenarios, metadata, _ = suite.build_confirmation_scenarios(u)
    pool = json.loads((source / "memory_pool.json").read_text())
    manifest = json.loads((source /
        "fast_v2_confirmation_seed_2603/agent_manifest.json").read_text())
    agent = RemoteJSONAgent(args.output, args.seed, env=args.env,
        tokenizer_path=Path(manifest["tokenizer_path"]), temperature=0.0,
        max_new_tokens=384, workers=4, exact_prompt_cache=False)

    selected = sorted(name for name, spec in metadata.items()
                      if spec["family"].startswith("literal_") or
                      spec["family"] == "latest_branch_contact")
    rows = []
    original = suite.v2
    suite.v2 = v3
    try:
        for task in selected:
            query, scenario = metadata[task]["public_request"], scenarios[task]
            began = perf_counter_ns()
            bank = v3.fast_select(pool, query, scenario, agent.tokenizer, adaptive=True)
            selection_ms = (perf_counter_ns() - began) / 1e6
            job = initial_job_v2(task, scenario, bank, u, agent.tokenizer,
                                 "fast_v3_development", "Contract-Adaptive-v2")
            job["selection_latency_ms"] = selection_ms
            row = suite.run_batch([job], {task: query}, agent, args.output, u)[0]
            remote, local, accounted = suite._latency(row)
            item = suite.annotate_direct({**row,
                "method": "Contract-Adaptive-v3", "actor_seed": args.seed,
                "trajectory": metadata[task]["family"], "session_index": 0,
                "selected_bank": [note["uid"] for note in bank],
                "remote_latency_ms": remote, "local_runtime_ms": local,
                "accounted_latency_ms": accounted,
                "public_request_sha256": suite._sha(query)}, metadata)
            rows.append(item)
    finally:
        suite.v2 = original
        agent.close()

    with (args.output / "sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"development_only": True, "disclosed_tasks": True,
        "seed": args.seed, "sessions": len(rows),
        "requested_state_successes": sum(row["requested_state_success"] for row in rows),
        "process_state_successes": sum(row["process_state_success"] for row in rows),
        "live_remote_requests": agent.calls,
        "mean_accounted_latency_ms": sum(row["accounted_latency_ms"] for row in rows) /
                                      len(rows)}
    adapter.write(args.output, "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
