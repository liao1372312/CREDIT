"""Fresh-seed paired comparison of frozen v1 and corrected mechanism fits.

The teacher policy, task definitions, API behavior, retrieval, budgets and all
observations are identical. This isolates kernel/training corrections; it does
not turn the restricted workload into the paper's structured agent benchmark.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from . import shared_experiments
from .llm_policy import install_policy
from .stateful import StateStep, StateTask, StateTrace
from .stateful_experiments import _json
from .paired_statistics import summarize
from .supported_mechanisms import fit_supported


def load_traces(path):
    return [StateTrace(task=StateTask(**{**r["task"], "roles": tuple(r["task"]["roles"])}),
        exposure=tuple(r["exposure"]), steps=tuple(StateStep(**s) for s in r["steps"]),
        outcome=r["outcome"], checkpoint_hash=r["checkpoint_hash"], state_hash=r["state_hash"],
        calls=r["calls"], reused=r.get("reused", 0)) for r in json.loads(path.read_text())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mechanism_correction_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("results/mechanism_correction_v2"))
    parser.add_argument("--policy", type=Path, default=Path("results/llm_policy"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory to preserve experiments")
    config = json.loads(args.config.read_text())
    backend = json.loads((args.policy / "model_manifest.json").read_text())
    if sha256((args.policy / "decision_scores.json").read_bytes()).hexdigest() != backend["decision_scores_sha256"]:
        raise ValueError("policy checksum mismatch")
    install_policy(args.policy / "decision_scores.json")
    args.output.mkdir(parents=True)
    sources = ("supported_mechanisms.py", "mechanism_correction_experiments.py", "shared_mechanisms.py",
               "shared_experiments.py", "stateful.py", "mechanisms.py", "prospective.py",
               "stateful_experiments.py", "llm_policy.py", "paired_statistics.py")
    digests = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
    _json(args.output / "frozen_protocol.json", {"config": config, "executed_seeds": config["seeds"],
        "config_sha256": sha256(args.config.read_bytes()).hexdigest(), "source_sha256": digests,
        "backend": backend, "policy_directory": str(args.policy.resolve()),
        "comparison": "same data and heldout tasks: v1 fixed80 vs public-mutation support, bounded interactions, validation checkpoints",
        "selection": "conditional NLL on calibration trajectories; no heldout labels or private probabilities",
        "limits": "old workload retained intentionally; workload adequacy is a separate unresolved issue"})
    (args.output / "source").mkdir()
    for name in sources:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    old_fitter = shared_experiments.fit_shared
    all_rows, fit_rows = [], []
    for seed in config["seeds"]:
        seed_rows = {}
        for version in ("corrected", "original"):
            directory = args.output / version / f"seed_{seed}"
            selected = []
            def corrected_fitter(*arguments, **keywords):
                validation = load_traces(directory / "calibration_traces.json")
                model = fit_supported(*arguments, **keywords, validation_traces=validation,
                                      patience=config["validation_patience"])
                selected.append({"selected_epoch": model.selected_epoch,
                    "executed_epochs": len(model.training_history), "best_validation_nll": model.best_validation_nll})
                return model
            shared_experiments.fit_shared = corrected_fitter if version == "corrected" else old_fitter
            fitting_config = {**config, "epochs": config["epochs"] if version == "corrected" else config["original_epochs"]}
            try:
                episodes, report = shared_experiments.run_seed(fitting_config, seed, directory)
            finally:
                shared_experiments.fit_shared = old_fitter
            report["scope"] = "restricted Qwen teacher / synthetic v1 API; mechanism correction diagnostic"
            report["fit_version"] = version
            if selected:
                report["validation_checkpoints"] = selected
                report["optimizer"] = "Adam/proximal groups, bounded interactions, public response support, conditional-NLL checkpoint selection"
            _json(directory / "report.json", report)
            seed_rows[version] = episodes
            fit_rows.extend({"seed": seed, "version": version, "model": index, **row} for index, row in enumerate(selected))
            all_rows.extend({**row, "method": f"{version}/{row['method']}"} for row in episodes)
            print(json.dumps({"seed": seed, "version": version, "complete": True}), flush=True)
        for name in ("decision_observations.json", "api_observations.json", "training_traces.json", "calibration_traces.json", "split_manifest.json"):
            left = args.output / "corrected" / f"seed_{seed}" / name
            right = args.output / "original" / f"seed_{seed}" / name
            if left.read_bytes() != right.read_bytes():
                raise AssertionError(f"comparison data differs: {name}")
    table, comparisons = summarize(all_rows, "corrected/CREDIT-empirical")
    from .experiments import _write_csv
    _write_csv(args.output / "pooled_main.csv", table)
    _write_csv(args.output / "pooled_comparisons.csv", comparisons)
    _write_csv(args.output / "checkpoints.csv", fit_rows)
    _json(args.output / "summary.json", {"main": table,
        "comparison_inputs_byte_identical": True,
        "scope": "kernel correction diagnostic; no claim of paper reproduction or benchmark improvement"})


if __name__ == "__main__":
    main()
