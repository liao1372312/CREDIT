"""Reexecute a frozen v3 local cycle and reconstruct effects/PPI decisions.

Neural weights are checked against frozen digests and evaluated, not retrained.
No measured outcome can modify a bank, model or gate during this audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from .drift import Level, ProbeEvidence, TransportGate, transfer_level
from .experiment import digest, write
from .factorial import eligible_slots, estimate, regular_fraction, admitted_pairs
from .inference import certify, ppi, tune_lambda, worst_mixture
from .local import StatefulExecutor, factory, memories, ranked, stable_seed, differential_probes
from .outcome import OutcomeModel
from .selection import BankSelector, Ledger, bank_cost
from .types import Context, Node, Role, SideEffect


def load_context(data):
    nodes = []
    for n in data["nodes"]:
        r = n["role"]
        role = Role(tuple(r["inputs"]), tuple(r["consumed_outputs"]), SideEffect(r["side_effect"]), r["authorization"])
        nodes.append(Node(n["uid"], role, n["version"], tuple(n["features"])))
    return Context(data["uid"], data["stratum"], tuple(nodes), tuple(tuple(e) for e in data["edges"]),
                   data["seed"], data["initial_state"])


class SavedCalibration:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def predict(self, values):
        return np.interp(values, self.x, self.y)


def check_close(a, b, where):
    if not np.allclose(a, b, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{where} differs")


def audit(output):
    torch.set_num_threads(1)
    read = lambda name: json.loads((output / name).read_text())
    manifest = read("artifact_manifest.json")
    for name, expected in manifest.items():
        if digest(output / name) != expected:
            raise ValueError(f"frozen artifact differs: {name}")
    protocol = read("frozen_protocol.json")
    if digest(protocol["paper"]) != protocol["paper_sha256"]:
        raise ValueError("the manuscript changed after the frozen run")
    for name, expected in protocol["source_sha256"].items():
        if digest(name) != expected or digest(output / "source" / Path(name).name) != expected:
            raise ValueError(f"source differs: {name}; audit with the frozen source snapshot")
    config = protocol["config"]
    executor, pool = StatefulExecutor(), memories()
    records = [json.loads(line) for line in (output / "raw_episodes.jsonl").read_text().splitlines()]
    references, rows, contexts, observed, costs = {}, defaultdict(list), {}, {}, defaultdict(lambda: [0, 0])
    restored = 0
    for r in records:
        data, phase = r["episode"], r["phase"]
        c = load_context(data["context"])
        a = tuple(data["exposure"])
        contexts[c.uid] = c
        if phase in {"factorial", "calibration"} and c.uid in references:
            e = executor.replay(c, a, references[c.uid])
        else:
            e = executor.run(c, a)
            if phase in {"factorial", "calibration"}:
                references[c.uid] = e
        if (e.state_hash != data["state_hash"] or e.checkpoint_hash != data["checkpoint_hash"]
                or e.restored_hash != data["restored_hash"] or e.verified != data["verified"]
                or e.calls != data["calls"] or e.reused_calls != data["reused_calls"]
                or list(e.events) != data["events"]
                or [asdict(o) for o in e.observations] != data["observations"] or not e.replay_valid):
            raise ValueError(f"actual execution differs: {phase}, {c.uid}")
        costs[phase][0] += 1
        costs[phase][1] += e.calls
        restored += int(e.restored_hash == e.checkpoint_hash)
        rows[phase].append((c, a, e.outcome, data))
        key = c.uid, a
        if key in observed and observed[key] != e.outcome:
            raise ValueError("CRN-paired repeated execution is inconsistent")
        observed[key] = e.outcome
    for phase, (n, calls) in costs.items():
        if n != read("costs.json")["episodes"][phase] or calls != read("costs.json")["api_calls"][phase]:
            raise ValueError("budget ledger differs")
        if n > read("costs.json")["episode_limits"][phase]:
            raise ValueError("budget exceeded")
    # Reconstruct actual randomized slots/rows from the recorded RNG and checkpoint.
    design_count = 0
    for phase, file, seed, draws in [("factorial", "factorial_estimates.json", config["seed"], config["draws"]),
                                      ("calibration", "calibration_designs.json", config["seed"] + 1, 1)]:
        rng = np.random.default_rng(seed)
        for report in read(file):
            c = load_context(report["context"])
            panel = eligible_slots(c, pool, ranked(c, tuple(m.uid for m in pool)), 4, 2, rng)
            if [m.uid for m in panel] != report["memory_ids"]:
                raise ValueError("reserved-slot sampling differs")
            d = regular_fraction(len(panel), admitted_pairs(c, panel), seed=stable_seed(c.uid))
            assignments, outcomes = [], []
            for block in range(draws):
                z = d.draw(rng)
                recorded = [r for r in rows[phase] if r[0].uid == f"{c.uid}:draw-{block}"]
                if len(recorded) != len(z):
                    raise ValueError("incomplete fractional draw")
                y = []
                for row, (task, exposure, outcome, data) in zip(z, recorded):
                    expected = tuple(m.uid for m, sign in zip(panel, row) if sign == 1)
                    if exposure != expected or data["assignment_probability"] != 2**-len(panel) or not data["randomized"]:
                        raise ValueError("assignment/propensity differs")
                    if task.seed != stable_seed(c.seed, block):
                        raise ValueError("draw noise is not checkpoint-paired")
                    y.append(outcome)
                assignments.append(z)
                outcomes.append(y)
            result = estimate(d, assignments, outcomes)
            check_close(result["beta"], report["beta"], "factorial coefficients")
            check_close(result["banzhaf"], report["banzhaf"], "Banzhaf credit")
            if draws > 1:
                check_close(result["draw_standard_error"], report["draw_standard_error"], "draw-cluster uncertainty")
            design_count += 1
    support = defaultdict(int)
    for c, a, y, data in rows["factorial"]:
        nodes = {n.uid: n for n in c.nodes}
        for o in data["observations"]:
            if o["label"] in {"ok", "agent"}:
                n = nodes[o["node"]]
                support[(n.role.key, n.version)] += 1
    evidence, fixtures = differential_probes(seed=config["seed"], support=support, fixtures=config["fixtures"])
    if [asdict(p) for p in evidence] != read("transfer_levels.json") or fixtures != read("differential_fixtures.json"):
        raise ValueError("differential fixtures/Clopper-Pearson levels differ")
    gate = TransportGate(evidence)
    mixture = {k: 1 / len(protocol["strata"]) for k in protocol["strata"]}
    models = []
    for path in sorted(output.glob("outcome_model_*.pt")):
        # Only deserialize this run's own weights AFTER matching the frozen digest.
        data = torch.load(path, map_location="cpu", weights_only=False)
        model = OutcomeModel(pool, [r[0] for r in rows["factorial"]])
        if model.roles != data["role_indices"] or [m.uid for m in pool] != data["memory_ids"]:
            raise ValueError("model role/memory encoder metadata differs")
        model.heads.load_state_dict(data["state_dict"])
        model.calibrator = SavedCalibration(data["calibration_x"], data["calibration_y"])
        model.fit_report = data["fit_report"]
        calibration = rows["calibration"]
        raw = model.predict_many([(c, a) for c, a, _, _ in calibration], calibrated=False)
        refit = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, [r[2] for r in calibration])
        check_close(refit.X_thresholds_, data["calibration_x"], "held-out calibration X")
        check_close(refit.y_thresholds_, data["calibration_y"], "held-out calibration Y")
        models.append(model)
    from .local import retriever
    selector = BankSelector(pool, models, gate, retriever(config["exposure_depth"]), factory(config["seed"]),
                            executor, Ledger({}), mixture=mixture, capacity=config["capacity"],
                            cost_penalty=config["cost_penalty"])
    certificate_count = 0
    freeze_path = output / "certification_freeze.json"
    if freeze_path.exists():
        freeze = read("certification_freeze.json")
        incumbent = tuple(freeze["incumbent"])
        finalists = [tuple(b) for b in freeze["finalists"]]
        estimates_by_bank = {}
        for bank in [incumbent] + finalists:
            for k, predictions in freeze["prediction_hash_input"]["|".join(bank)].items():
                for phase, key, n in [("pilot", "pilot", config["pilot_n"]),
                                       ("certification", "labeled", config["certification_n"]),
                                       ("certification-unlabeled", "unlabeled", config["unlabeled_n"])]:
                    tasks = [selector.factory(k, phase, i) for i in range(n)]
                    check_close(selector.predictions(tasks, bank), predictions[key], "frozen PPI prediction")
        actual_certificates = read("switch_decision.json")["certificates"]
        for b, recorded in zip(finalists, actual_certificates):
            estimates = {}
            for k in mixture:
                pilot_tasks = [selector.factory(k, "pilot", i) for i in range(config["pilot_n"])]
                cert_tasks = [selector.factory(k, "certification", i) for i in range(config["certification_n"])]
                delta = lambda tasks: np.asarray([observed[(c.uid, selector.exposures(c, b))]
                                                  - observed[(c.uid, selector.exposures(c, incumbent))] for c in tasks])
                pred = freeze["prediction_hash_input"]["|".join(b)][k]
                base = freeze["prediction_hash_input"]["|".join(incumbent)][k]
                transported = all(gate.transported(c) for c in cert_tasks + pilot_tasks)
                weight = tune_lambda(delta(pilot_tasks), np.asarray(pred["pilot"]) - base["pilot"],
                                     labeled_n=config["certification_n"], unlabeled_n=config["unlabeled_n"],
                                     transported=transported)
                check_close(weight, freeze["pilot_weights"]["|".join(b)][k], "independent pilot weight")
                e = ppi(delta(cert_tasks), np.asarray(pred["labeled"]) - base["labeled"],
                        np.asarray(pred["unlabeled"]) - base["unlabeled"], weight)
                estimates[k] = e
                for field, value in asdict(e).items():
                    check_close(value, recorded["estimates"][k][field], f"PPI {field}")
            expected = certify(estimates, mixture,
                               cost_difference=bank_cost(b, pool) - bank_cost(incumbent, pool),
                               cost_penalty=config["cost_penalty"], finalists=len(finalists))
            if expected["passed"] != recorded["passed"]:
                raise ValueError("switch certification differs")
            check_close(expected["aggregate_lower"], recorded["aggregate_lower"], "aggregate interval")
            for k in mixture:
                check_close(expected["stratum_lower"][k], recorded["stratum_lower"][k], "noninferiority guard")
            certificate_count += 1
        accepted = [r for r in actual_certificates if r["passed"]]
        best = max(accepted, key=lambda c: c["aggregate_lower"] - c["cost_threshold"]) if accepted else None
        if list(best["bank"] if best else incumbent) != read("switch_decision.json")["bank"]:
            raise ValueError("adopted bank differs from certified finalist")
    report = {"verified": True, "actual_episodes_reexecuted": len(records),
              "api_calls_reexecuted": sum(v[1] for v in costs.values()), "restoration_hashes_checked": restored,
              "factorial_context_estimates_recomputed": design_count,
              "differential_fixture_calls_reexecuted": sum(f["api_calls"] for f in fixtures),
              "certificates_reconstructed": certificate_count,
              "model_weights": "frozen hashes and predictions checked; optimizer not retrained",
              "scope": protocol["scope"], "auditor_sha256": digest(__file__)}
    write(output / "execution_audit.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    audit(parser.parse_args().results)


if __name__ == "__main__":
    main()
