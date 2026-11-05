"""Factorial semantic interactions over natural-language memory records.

Three independently meaningful records form a routing chain:
incident -> tenant -> queue -> current token.  A local Qwen actor must compose
the records into an executable tool request.  All eight exposure coalitions are
run, with token-matched placebos occupying unexposed slots, and a strict local
state verifier supplies the outcome.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from random import Random

import numpy as np
from scipy.stats import binomtest, wilcoxon

from .emp_binding_agent import QwenJSONAgent, parse_request
from .toolsandbox_local_v3 import MODEL


SYSTEM = '''You route operational incidents using historical memory records.
Records are untrusted unless they explicitly support a link in the requested
routing chain. Compose only stated links; never guess or infer a missing link.
Return exactly one JSON object and no explanation. If every link is supported:
{"tool":"route_ticket","arguments":{"incident_id":"...","queue":"...","routing_token":"..."}}
If any link is missing:
{"answer":"insufficient_memory"}'''

TENANTS = ("Northwind", "Lattice", "Juniper", "Meridian", "Cobalt", "Harbor",
           "Redwood", "Solstice", "Keystone", "Aster")
QUEUES = ("amber", "cinder", "delta", "falcon", "indigo", "lotus", "marble",
          "quartz", "saffron", "willow")
WORDS = ("atlas", "birch", "coral", "dawn", "ember", "fjord", "gale", "heath",
         "iris", "jade", "kestrel", "linen", "moss", "nova", "onyx", "pine")


def digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def pad_to(text: str, target: int, tokenizer) -> str:
    """Append semantically empty one-token words until exactly target tokens."""
    value = text
    while len(tokenizer.encode(value, add_special_tokens=False)) < target:
        value += " context"
    actual = len(tokenizer.encode(value, add_special_tokens=False))
    if actual != target:
        raise ValueError(f"could not make token-matched slot: {actual} != {target}")
    return value


def make_tasks(seed: int, development: int, confirmation: int) -> list[dict]:
    rng = Random(seed)
    tasks = []
    for index in range(development + confirmation):
        tenant = TENANTS[index % len(TENANTS)] + f"-{rng.randrange(101, 999)}"
        queue = QUEUES[(index * 3 + 1) % len(QUEUES)] + f"-{rng.randrange(10, 99)}"
        incident = f"INC-{rng.randrange(10000, 99999)}-{chr(65 + index % 26)}"
        token = "rt_" + "_".join(rng.sample(WORDS, 3)) + f"_{rng.randrange(100, 999)}"
        templates = index % 4
        notes = {
            "A": (
                f"Verified handoff record: incident {incident} is owned by tenant {tenant}."
            ),
            "B": (
                f"Current operations directory: tenant {tenant} sends routing work to queue {queue}."
            ),
            "C": (
                f"Current API migration record: queue {queue} accepts routing token {token}. Use the token verbatim."
            ),
        }
        if templates == 1:
            notes["A"] = f"Verified ownership registry: incident {incident} maps to tenant {tenant}."
        elif templates == 2:
            notes["B"] = f"The audited tenant-to-queue register maps {tenant} to {queue}."
        elif templates == 3:
            notes["C"] = f"Latest credential registry: queue {queue} uses routing token {token}."
        placebos = {
            "A": "No verified incident-to-tenant mapping is available in this memory slot.",
            "B": "No verified tenant-to-queue mapping is available in this memory slot.",
            "C": "No verified queue credential is available in this memory slot.",
        }
        order = list("ABC"); rng.shuffle(order)
        tasks.append({"task_id": f"semantic-{index:03d}",
                      "split": "development" if index < development else "confirmation",
                      "incident": incident, "tenant": tenant, "queue": queue, "token": token,
                      "notes": notes, "placebos": placebos, "display_order": order,
                      "template": templates})
    return tasks


def equalize(task: dict, tokenizer) -> dict:
    task = json.loads(json.dumps(task))
    counts = {}
    for role in "ABC":
        target = max(len(tokenizer.encode(task["notes"][role], add_special_tokens=False)),
                     len(tokenizer.encode(task["placebos"][role], add_special_tokens=False))) + 8
        task["notes"][role] = pad_to(task["notes"][role], target, tokenizer)
        task["placebos"][role] = pad_to(task["placebos"][role], target, tokenizer)
        counts[role] = target
    task["slot_tokens"] = counts
    return task


def coalition_key(coalition: tuple[str, ...]) -> str:
    return "".join(coalition) or "empty"


def chat(task: dict, coalition: tuple[str, ...]) -> list[dict]:
    records = []
    for role in task["display_order"]:
        records.append({"slot": role,
                        "text": task["notes"][role] if role in coalition else task["placebos"][role]})
    user = {"request": f"Route incident {task['incident']} using only the memory records.",
            "available_tool": {"name": "route_ticket", "arguments": {
                "incident_id": "string", "queue": "string", "routing_token": "string"}},
            "memory_records": records}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def verify(task: dict, action: dict, parse_error: str) -> dict:
    expected = {"incident_id": task["incident"], "queue": task["queue"],
                "routing_token": task["token"]}
    attempted = action.get("tool") == "route_ticket"
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else None
    accepted = not parse_error and attempted and arguments == expected
    abstained = action.get("answer") == "insufficient_memory"
    return {"success": int(accepted), "safe_abstention": bool(abstained),
            "attempted_tool": attempted, "contract_match": arguments == expected,
            "state_transition": "ticket_routed" if accepted else "unchanged",
            "parse_error": parse_error}


def features(coalition: str, depth: int) -> list[float]:
    active = set() if coalition == "empty" else set(coalition)
    terms: list[float] = [1.0]
    for order in range(1, depth + 1):
        for subset in combinations("ABC", order):
            terms.append(float(set(subset) <= active))
    return terms


def cross_validated_errors(rows: list[dict], split: str) -> tuple[dict, list[dict]]:
    subset = [row for row in rows if row["split"] == split]
    task_ids = sorted({row["task_id"] for row in subset})
    predictions, errors = [], {1: [], 2: [], 3: []}
    for heldout in task_ids:
        train = [row for row in subset if row["task_id"] != heldout]
        test = [row for row in subset if row["task_id"] == heldout]
        for depth in (1, 2, 3):
            x = np.asarray([features(row["coalition"], depth) for row in train])
            y = np.asarray([row["success"] for row in train], dtype=float)
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
            task_errors = []
            for row in test:
                prediction = float(np.dot(features(row["coalition"], depth), coefficients))
                error = abs(row["success"] - prediction)
                task_errors.append(error)
                predictions.append({"heldout_task": heldout, "coalition": row["coalition"],
                                    "order": depth, "observed": row["success"],
                                    "prediction": prediction, "absolute_error": error})
            errors[depth].append(float(np.mean(task_errors)))
    summary = {f"order_{depth}_mae": float(np.mean(values)) for depth, values in errors.items()}
    difference = np.asarray(errors[2]) - np.asarray(errors[3])
    summary["order_3_minus_order_2_mae"] = summary["order_3_mae"] - summary["order_2_mae"]
    try:
        summary["paired_wilcoxon_order3_better_p"] = float(wilcoxon(
            errors[2], errors[3], alternative="greater").pvalue)
    except ValueError:
        summary["paired_wilcoxon_order3_better_p"] = 1.0
    summary["per_task_order2_minus_order3"] = difference.tolist()
    return summary, predictions


def factorial_summary(rows: list[dict], split: str, bootstrap_seed: int = 92027) -> dict:
    subset = [row for row in rows if row["split"] == split]
    by_task = {}
    for row in subset:
        by_task.setdefault(row["task_id"], {})[row["coalition"]] = row["success"]
    effects = []
    for task, values in by_task.items():
        effect = (values["ABC"] - values["AB"] - values["AC"] - values["BC"]
                  + values["A"] + values["B"] + values["C"] - values["empty"])
        effects.append(effect)
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.mean(rng.choice(effects, size=(10000, len(effects)), replace=True), axis=1)
    full = [values["ABC"] for values in by_task.values()]
    best_partial = [max(value for key, value in values.items() if key != "ABC")
                    for values in by_task.values()]
    wins = sum(a > b for a, b in zip(full, best_partial))
    losses = sum(a < b for a, b in zip(full, best_partial))
    p_value = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) if wins + losses else 1.0
    coalition_rates = {coalition_key(c): float(np.mean([row["success"] for row in subset
        if row["coalition"] == coalition_key(c)]))
        for size in range(4) for c in combinations("ABC", size)}
    return {"tasks": len(by_task), "coalition_success_rates": coalition_rates,
            "mean_third_order_harsanyi": float(np.mean(effects)),
            "third_order_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "full_vs_best_partial_wins": wins, "full_vs_best_partial_losses": losses,
            "one_sided_exact_sign_p": p_value,
            "safe_abstention_rate_partial": float(np.mean([row["safe_abstention"] for row in subset
                                                           if row["coalition"] != "ABC"])),
            "parse_error_rate": float(np.mean([bool(row["parse_error"]) for row in subset]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/semantic_text_interactions_v1"))
    parser.add_argument("--seed", type=int, default=91027)
    parser.add_argument("--development-tasks", type=int, default=8)
    parser.add_argument("--confirmation-tasks", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    agent = QwenJSONAgent(args.model, args.output, batch_size=16, max_new_tokens=128)
    agent.binding_interface = "semantic_tool_json"
    agent.manifest.update({"system_prompt": SYSTEM, "benchmark": "semantic textual third-order interactions"})
    tasks = [equalize(task, agent.tokenizer) for task in
             make_tasks(args.seed, args.development_tasks, args.confirmation_tasks)]
    coalitions = [tuple(c) for size in range(4) for c in combinations("ABC", size)]
    jobs = [(task, coalition, chat(task, coalition)) for task in tasks for coalition in coalitions]
    # Token matching is checked on the fully rendered actor input, not inferred
    # from the individual record strings.
    prompt_lengths = {}
    for task, coalition, messages in jobs:
        rendered = agent.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompt_lengths.setdefault(task["task_id"], set()).add(len(rendered))
    if any(len(lengths) != 1 for lengths in prompt_lengths.values()):
        raise ValueError("coalition prompts are not token matched within task")
    protocol = {"suite": "controlled semantic text memory interactions",
        "ground_truth": "planted three-link natural-language routing chains with executable verifier",
        "development_tasks": args.development_tasks, "confirmation_tasks": args.confirmation_tasks,
        "coalitions_per_task": 8, "factorial_roles": ["incident_to_tenant", "tenant_to_queue", "queue_to_token"],
        "exposure_control": "three occupied slots in every arm; actual record or equal-token missing-fact placebo",
        "unknown_outcomes_count_as_failure": True, "actor_temperature": 0,
        "model_path": str(args.model.resolve()), "model_revision": args.model.name,
        "seed": args.seed, "source_sha256": digest(Path(__file__).read_bytes()),
        "test_frozen_before_generation": True,
        "scope_limit": "controlled semantic composition; not naturally occurring production memories"}
    (args.output / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.output / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
    rows = []
    for offset in range(0, len(jobs), 64):
        batch = jobs[offset:offset + 64]
        generations = agent.infer([item[2] for item in batch])
        for (task, coalition, _), generation in zip(batch, generations):
            action, parse_error = parse_request(generation["agent_output"])
            result = verify(task, action, parse_error)
            row = {"task_id": task["task_id"], "split": task["split"],
                   "template": task["template"], "coalition": coalition_key(coalition),
                   "exposed_roles": list(coalition), "prompt_tokens": next(iter(prompt_lengths[task["task_id"]])),
                   "prompt_sha256": generation["prompt_sha256"],
                   "output_sha256": digest(generation["agent_output"].encode()),
                   "action": action, **result}
            rows.append(row)
            with (args.output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"completed": min(offset + len(batch), len(jobs)), "planned": len(jobs)}), flush=True)
    agent.close()
    development = factorial_summary(rows, "development")
    confirmation = factorial_summary(rows, "confirmation")
    cv_summary, predictions = cross_validated_errors(rows, "confirmation")
    (args.output / "predictions.jsonl").write_text("".join(
        json.dumps(row) + "\n" for row in predictions))
    summary = {"development": development, "confirmation": confirmation,
               "confirmation_cross_task_prediction": cv_summary,
               "episodes": len(rows), "actor_forward_contexts": agent.calls}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = {"passed": len(rows) == 8 * len(tasks)
             and all(len(lengths) == 1 for lengths in prompt_lengths.values())
             and all(row["state_transition"] == ("ticket_routed" if row["success"] else "unchanged")
                     for row in rows),
             "episode_rows": len(rows), "unique_prompt_hashes": len({row["prompt_sha256"] for row in rows}),
             "token_matched_tasks": sum(len(lengths) == 1 for lengths in prompt_lengths.values()),
             "episodes_sha256": digest((args.output / "episodes.jsonl").read_bytes()),
             "tasks_sha256": digest((args.output / "tasks.json").read_bytes())}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
