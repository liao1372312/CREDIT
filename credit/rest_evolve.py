"""Local HTTP snapshot suite with an injected, unchanged-schema submit drift."""

from __future__ import annotations

import argparse
import json
import threading
import time
from hashlib import sha256
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from random import Random
from statistics import mean
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .experiments import _write_csv
from .policy import retrieve
from .sandbox import Task, _decision_probability, _uniform, make_tasks


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.version = "v1"
        self.task_id = ""
        self.required: tuple[str, ...] = ()
        self.completed: list[str] = []
        self.submitted = False
        self.forbidden_writes = 0

    def reset(self, task: Task) -> None:
        with self.lock:
            self.version = task.version
            self.task_id = task.uid
            self.required = task.roles
            self.completed = []
            self.submitted = False
            self.forbidden_writes = 0


def _handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _reply(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _process(self, body: dict) -> None:
            role = self.path.strip("/")
            with state.lock:
                if role not in ("lookup", "transform", "submit"):
                    self._reply(404, {"error": "unknown operation"})
                    return
                if role == "lookup":
                    valid = body.get("key") == state.task_id
                elif role == "transform":
                    valid = (body.get("rule") == "canonical"
                             and body.get("value") == state.task_id)
                else:
                    required = set(state.required) - {"submit"}
                    valid = (body.get("token") == state.task_id
                             and body.get("contract") == state.version
                             and required <= set(state.completed))
                if not valid:
                    self._reply(400, {"ok": False, "error": "contract rejected"})
                    return
                state.completed.append(role)
                if role == "submit":
                    state.submitted = True
                self._reply(200, {"ok": True, "value": state.task_id})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._reply(400, {"ok": False, "error": "invalid JSON"})
                return
            self._process(body)

    return Handler


@contextmanager
def local_service():
    state = _State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_rest_task(task: Task, exposure: tuple[str, ...],
                  state: _State, base_url: str) -> dict:
    state.reset(task)
    start = time.perf_counter()
    calls = 0
    for index, role in enumerate(task.roles):
        correct = _uniform(task.seed, index, "decision") < _decision_probability(
            role, task.version, frozenset(exposure), task.drift_intensity
        )
        if role == "lookup":
            body = {"key": task.uid if correct else "wrong"}
        elif role == "transform":
            body = {"value": task.uid, "rule": "canonical" if correct else "legacy"}
        elif role == "submit":
            body = {"token": task.uid,
                    "contract": task.version if correct else
                    ("v2" if task.version == "v1" else "v1")}
        else:
            raise ValueError(f"unsupported REST operation: {role}")
        request = Request(f"{base_url}/{role}", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        calls += 1
        try:
            with urlopen(request, timeout=5) as response:
                success = response.status == 200
        except HTTPError:
            success = False
        if not success:
            break
    with state.lock:
        verified = (state.submitted and state.forbidden_writes == 0
                    and set(task.roles) <= set(state.completed))
    return {
        "success": int(verified), "api_calls": calls,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


def run_http_suite(bank_report: Path, output: Path, *, seed: int = 17,
                   tasks_per_version: int = 40) -> dict:
    banks = json.loads(bank_report.read_text(encoding="utf-8"))["banks"]
    tasks = (
        make_tasks(tasks_per_version, split="rest-unseen", version="v1",
                   templates=("new_order", "long"), seed=seed)
        + make_tasks(tasks_per_version, split="rest-drift", version="v2",
                     templates=("new_order", "long"), seed=seed + 1)
    )
    rows = []
    with local_service() as (state, base_url):
        for method, bank in banks.items():
            for task in tasks:
                exposure = retrieve(task, bank, depth=4)
                result = run_rest_task(task, exposure, state, base_url)
                rows.append({
                    "method": method, "task": task.uid, "family": task.family,
                    "version": task.version, "template": task.template,
                    "success": result["success"], "api_calls": result["api_calls"],
                    "latency_ms": result["latency_ms"],
                    "exposure": "|".join(exposure),
                })
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "http_episodes.csv", rows)
    summary = [{
        "method": method, "version": version,
        "n": sum(r["method"] == method and r["version"] == version for r in rows),
        "vsr": mean(r["success"] for r in rows
                    if r["method"] == method and r["version"] == version),
        "mean_api_calls": mean(r["api_calls"] for r in rows
                               if r["method"] == method and r["version"] == version),
        "mean_latency_ms": mean(r["latency_ms"] for r in rows
                                if r["method"] == method and r["version"] == version),
    } for method in banks for version in ("v1", "v2")]
    _write_csv(output / "http_summary.csv", summary)
    comparisons = []
    rng = Random(seed + 2027)
    for version in ("v1", "v2"):
        credit = {r["task"]: r for r in rows
                  if r["method"] == "CREDIT" and r["version"] == version}
        fifo = {r["task"]: r for r in rows
                if r["method"] == "FIFO" and r["version"] == version}
        families = sorted({r["family"] for r in credit.values()})
        by_family = {
            family: [credit[task]["success"] - fifo[task]["success"]
                     for task in credit if credit[task]["family"] == family]
            for family in families
        }
        observed = mean(value for values in by_family.values() for value in values)
        boot = [mean(value for family in (rng.choice(families) for _ in families)
                     for value in by_family[family]) for _ in range(1000)]
        comparisons.append({
            "version": version, "baseline": "FIFO",
            "paired_delta_vsr": observed,
            "cluster_ci_low": sorted(boot)[24],
            "cluster_ci_high": sorted(boot)[974],
            "n": len(credit),
        })
    (output / "http_comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    manifest = {
        "suite": "WebAPI-Evolve local injected-drift subset",
        "provider_drift": False,
        "source_version": "v1", "target_version": "v2",
        "change_type": "unchanged JSON schema; submit contract value changes",
        "version_fingerprints": {
            version: sha256(
                json.dumps({"submit_schema": ["token", "contract"],
                            "accepted_contract": version}, sort_keys=True).encode()
            ).hexdigest()
            for version in ("v1", "v2")
        },
        "endpoint_paths": ["/lookup", "/transform", "/submit"],
        "isolated_service": "Python ThreadingHTTPServer bound to 127.0.0.1",
        "verifier": "submit completed after all required roles; no forbidden writes",
        "tasks": len(tasks), "seed": seed,
        "bank_source": str(bank_report),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-report", type=Path,
                        default=Path("results/causalcompose/report.json"))
    parser.add_argument("--output", type=Path, default=Path("results/rest_evolve"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tasks-per-version", type=int, default=40)
    args = parser.parse_args()
    print(json.dumps(run_http_suite(args.bank_report, args.output, seed=args.seed,
                                    tasks_per_version=args.tasks_per_version), indent=2))


if __name__ == "__main__":
    main()
