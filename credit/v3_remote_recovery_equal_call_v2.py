"""Corrected equal-call protocol using the current online CREDIT selector.

Version 1 deliberately remains immutable: it diagnosed that the legacy
``v3_optimization.select`` surrogate search is not the selector specified by
the current paper.  This version changes only the CREDIT proposal/selection
path, uses fresh development and confirmation seeds, and retains the v1 direct
search, budget completion, held-out scoring, and audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .v3.experiment import digest, write
from . import v3_remote_recovery_equal_call as base


DEV_SEEDS = (67003,)
CONFIRM_SEEDS = (68023, 68041, 68053, 68059, 68071, 68087, 68099, 68111)
_BASE_FROZEN_PROTOCOL = base.frozen_protocol
_BASE_BUDGET_SELECT = base.budget_select


def frozen_protocol(policy, dataset, split):
    protocol = _BASE_FROZEN_PROTOCOL(policy, dataset, split)
    protocol.update({
        "experiment": "credit_v3_remote_recovery_equal_actual_selection_calls_v2",
        "credit_selector": "current v3_online_experiment mixed model/direct screening, physical halving, and fresh policy certification",
        "direct_selector": "budget-aware physical v3_optimization search retained from v1",
        "correction_from_v1": "v1 used the deprecated legacy optimizer for both arms; v2 restores the paper-current online CREDIT selector",
        "development_seed": list(DEV_SEEDS),
        "confirmation_seeds_frozen_before_v2_development_results": list(CONFIRM_SEEDS),
        "scope": "finite synthetic API and cached remote actor; shared evidence cost is outside the equal-call x-axis; v2 is a fresh corrected protocol",
    })
    protocol["source_sha256"] = protocol["source_sha256"] | {
        "credit/v3_remote_recovery_equal_call_v2.py": digest(__file__),
    }
    return protocol


def current_budget_select(models, gate, corpus, method, budget, directory):
    if method == "Direct":
        return _BASE_BUDGET_SELECT(models, gate, corpus, method, budget, directory)

    from . import v3_online_experiment as experiment
    experiment.CAP = budget
    state = {
        "policy": "covered",
        "models": models,
        "bank": (),
        "deployment_policy": experiment.policy_data(gate, ()),
    }
    ledger, _ = experiment.selection(state, gate, corpus, 0, directory)
    selected = tuple(state["bank"])
    # The common v1 integrity audit expects the physical selection stream under
    # this canonical filename.  Preserve the current selector's JSON artifacts.
    (directory / "executions.jsonl.gz").rename(directory / "selection.jsonl.gz")
    initial = int(ledger.report()["selection_api_calls_including_common"])
    candidates = base.frozen_candidates(directory, selected)
    bank, spent, group_bound, validation_episodes = base.complete_budget(
        corpus, gate, candidates, initial, budget, directory)
    write(directory / "final_selection.json", {
        "method": method, "budget": budget, "bank": list(bank),
        "bank_tokens": base.bank_cost(bank, base.RECOVERY_POOL),
        "candidate_banks_frozen_before_budget_validation": [list(value) for value in candidates],
        "actual_selection_calls": spent, "unspent": budget - spent,
        "next_complete_group_upper_bound": group_bound,
        "validation_episodes": validation_episodes,
        "selection_uses_heldout_labels": False,
        "selector": "current online CREDIT selector",
    })
    return bank, spent, group_bound


def install_v2():
    base.DEV_SEEDS = DEV_SEEDS
    base.CONFIRM_SEEDS = CONFIRM_SEEDS
    base.frozen_protocol = frozen_protocol
    base.budget_select = current_budget_select


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path,
                        default=Path("results/credit_v3_remote_llm_policy_v2"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--generate-dataset", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    install_v2()
    if args.generate_dataset:
        base.generate_dataset(args.dataset, args.split)
        write(args.dataset / "equal_call_v2_dataset_protocol.json", {
            "split": args.split, "seeds": list(base.seeds_for(args.split)),
            "correction": "fresh v2 seeds for the current online CREDIT selector",
            "wrapper_sha256": digest(__file__),
            "base_equal_call_sha256": digest("credit/v3_remote_recovery_equal_call.py"),
        })
        write(args.dataset / "manifest.json", {
            path.name: digest(path) for path in args.dataset.iterdir()
            if path.is_file() and path.name != "manifest.json"})
    elif args.audit:
        base.audit(args.policy, args.dataset, args.output, args.split)
    else:
        base.run(args.policy, args.dataset, args.output, args.split)
