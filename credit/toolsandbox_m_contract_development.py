"""Development-only check of execution contracts on disclosed failure tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_local_v3 import install_local_clock, run_batch, upstream_imports, write
from .toolsandbox_m_compose_confirmation import (
    CLOCK,
    annotate,
    build_confirmation_scenarios,
)
from .toolsandbox_m_compose_selector import compose_select
from .toolsandbox_m_contract_selector import contract_select
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order


METHODS = ("CREDIT-Compose", "CREDIT-Contract")
TASKS = (
    "compose_message_to_reminder_and_boss",
    "compose_message_to_reminder_and_boss_alt",
    "compose_replace_contact_and_notify_alt",
    "compose_replace_reminder_and_notify_alt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/toolsandbox_m_compose_confirmation_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/toolsandbox_m_contract_development_v1"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1201, 1213, 1223])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)

    protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    u = upstream_imports(Path(protocol["upstream_directory"]))
    install_local_clock(CLOCK)
    scenarios, metadata, _ = build_confirmation_scenarios(u)
    pool = json.loads((args.source / "memory_pool.json").read_text())
    manifest = next(args.source.glob("compose_confirmation_seed_*/agent_manifest.json"))
    tokenizer_path = Path(json.loads(manifest.read_text())["tokenizer_path"])
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True)

    selection = {}
    for task in TASKS:
        query = metadata[task]["public_request"]
        selection[task] = {}
        for method in METHODS:
            bank = (compose_select(pool, query, scenarios[task]) if method == "CREDIT-Compose"
                    else contract_select(pool, query, scenarios[task], tokenizer))
            selection[task][method] = [note["uid"] for note in bank]
    write(args.output, "selection.json", selection)

    rows = []
    for actor_index, seed in enumerate(args.seeds):
        directory = args.output / f"contract_development_seed_{seed}"
        directory.mkdir()
        agent = RemoteJSONAgent(directory, seed, env=args.env,
            tokenizer_path=tokenizer_path, temperature=0.0, max_new_tokens=384,
            workers=args.workers, exact_prompt_cache=False)
        agent.binding_interface = "tool_json"
        agent.manifest["benchmark"] = "ToolSandbox-M contract development"
        for task_index, task in enumerate(TASKS):
            query = metadata[task]["public_request"]
            order = balanced_order(list(METHODS), actor_index, task_index, 0)
            jobs, banks = [], {}
            for method in order:
                bank = (compose_select(pool, query, scenarios[task])
                        if method == "CREDIT-Compose" else
                        contract_select(pool, query, scenarios[task], agent.tokenizer))
                banks[method] = bank
                jobs.append(initial_job_v2(task, scenarios[task], bank, u,
                                            agent.tokenizer,
                                            "contract_development", method))
            for row in run_batch(jobs, agent, directory, u, max_calls=10):
                item = annotate({**row, "actor_seed": seed,
                    "trajectory": metadata[task]["family"], "session_index": 0,
                    "selected_bank": [note["uid"] for note in banks[row["method"]]],
                    "remote_latency_ms": session_latency(row)}, metadata)
                rows.append(item)
        agent.close()

    with (args.output / "sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        table.append({
            "method": method,
            "sessions": len(subset),
            "requested_state_VSR": float(np.mean([
                row["requested_state_success"] for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_latency_ms": float(np.mean([
                row["remote_latency_ms"] for row in subset])),
            "by_task": {task: float(np.mean([
                row["requested_state_success"] for row in subset if row["task"] == task
            ])) for task in TASKS},
        })
    write(args.output, "summary.json", {
        "development_only": True,
        "prior_confirmation_failures_examined": True,
        "table": table,
        "sessions": len(rows),
    })
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
