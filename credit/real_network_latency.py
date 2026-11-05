"""Measure a real read-only Web API plus remote LLM workflow.

Each trial fetches an official GitHub release record over a new HTTPS
connection, passes the returned release text to the configured OpenAI-compatible
LLM, and verifies a small semantic answer.  Credentials and endpoint URLs are
never written to the result directory.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .llm_remote_policy_v2 import load_config


RELEASES = (
    ("v2.14.1", False),
    ("v2.15.0", True),
)


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def github_call(tag: str, timeout: int) -> tuple[dict, dict]:
    url = f"https://api.github.com/repos/mccutchen/go-httpbin/releases/tags/{tag}"
    request = Request(url, headers={"Accept": "application/vnd.github+json",
                                    "User-Agent": "credit-network-latency"})
    began = time.perf_counter_ns()
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        status = response.status
        headers = {key.lower(): value for key, value in response.headers.items()
                   if key.lower() in {"x-ratelimit-limit", "x-ratelimit-remaining",
                                      "x-ratelimit-reset", "content-type"}}
    elapsed = (time.perf_counter_ns() - began) / 1e6
    payload = json.loads(raw)
    record = {"status": status, "latency_ms": elapsed,
              "response_sha256": digest_bytes(raw), "headers": headers,
              "response_bytes": len(raw)}
    return payload, record


def answer_value(content: object) -> bool | None:
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
        answer = value.get("trailers_added") if isinstance(value, dict) else None
        return answer if isinstance(answer, bool) else None
    except ValueError:
        match = re.search(r'"trailers_added"\s*:\s*(true|false)', text, re.I)
        return None if not match else match.group(1).lower() == "true"


def llm_call(config: dict, tag: str, release_body: str) -> tuple[dict, dict]:
    prompt = ("Below is the official upstream release text for go-httpbin " + tag + ".\n"
              + release_body[:7000] + "\n\nDid this release add a /trailers endpoint? "
              'Return exactly {"trailers_added":true} or {"trailers_added":false}.')
    body = json.dumps({"model": config["model"],
        "messages": [{"role": "system", "content": "Answer from the supplied release text only."},
                     {"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": min(32, config["max_tokens"])},
        separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = "Bearer " + config["api_key"]
    began = time.perf_counter_ns()
    request = Request(config["url"], data=body, headers=headers, method="POST")
    with urlopen(request, timeout=config["timeout"]) as response:
        raw = response.read()
        status = response.status
    elapsed = (time.perf_counter_ns() - began) / 1e6
    payload = json.loads(raw)
    choice = payload["choices"][0]
    content = choice.get("message", {}).get("content")
    record = {"status": status, "latency_ms": elapsed,
              "request_sha256": digest_bytes(body), "response_sha256": digest_bytes(raw),
              "response_bytes": len(raw), "parsed_answer": answer_value(content),
              "finish_reason": choice.get("finish_reason"), "usage": payload.get("usage", {})}
    return payload, record


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["workflow_success"]]
    warm = rows[1:]
    def metrics(field: str, subset: list[dict]) -> dict:
        values = [float(row[field]) for row in subset]
        return {"n": len(values), "mean_ms": statistics.fmean(values),
                "p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95),
                "min_ms": min(values), "max_ms": max(values)}
    return {
        "trials": len(rows), "workflow_successes": len(successful),
        "workflow_success_rate": len(successful) / len(rows),
        "external_https_calls": 2 * len(rows),
        "all_trials": {field: metrics(field, rows) for field in
                       ("tool_latency_ms", "llm_latency_ms", "end_to_end_latency_ms")},
        "warm_trials_excluding_first": {field: metrics(field, warm) for field in
                                        ("tool_latency_ms", "llm_latency_ms", "end_to_end_latency_ms")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("results/real_network_latency_v1"))
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    if args.trials < 10:
        raise ValueError("at least ten trials are required for a p95 estimate")
    config = load_config(args.env)
    args.output.mkdir(parents=True)
    protocol = {
        "scope": "real sequential HTTPS Web API plus remote LLM workflow",
        "collection_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trials": args.trials, "connections": "new urllib HTTPS request per call",
        "web_api": "GitHub REST releases endpoint", "web_api_authentication": "none",
        "llm_model": config["model"], "llm_base_url_sha256": config["base_url_sha256"],
        "llm_temperature": 0, "llm_max_output_tokens": min(32, config["max_tokens"]),
        "timeout_seconds": config["timeout"], "credentials_persisted": False,
        "timing_clock": "time.perf_counter_ns around complete urlopen/read",
        "cold_warm_policy": "report all trials and trials 2..N separately",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    with (args.output / "trials.jsonl").open("w") as stream:
        for index in range(args.trials):
            tag, expected = RELEASES[index % len(RELEASES)]
            total_began = time.perf_counter_ns()
            error = None
            tool = {"status": None, "latency_ms": float("nan")}
            llm = {"status": None, "latency_ms": float("nan"), "parsed_answer": None,
                   "usage": {}}
            try:
                release, tool = github_call(tag, config["timeout"])
                _, llm = llm_call(config, tag, release.get("body") or "")
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as exception:
                error = type(exception).__name__ + ": " + str(exception)[:300]
            total = (time.perf_counter_ns() - total_began) / 1e6
            row = {"trial": index, "tag": tag, "expected_answer": expected,
                   "tool_latency_ms": tool["latency_ms"], "llm_latency_ms": llm["latency_ms"],
                   "end_to_end_latency_ms": total,
                   "workflow_success": error is None and llm["parsed_answer"] is expected,
                   "error": error, "tool": tool, "llm": llm}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
            print(json.dumps({"trial": index + 1, "planned": args.trials,
                              "success": row["workflow_success"], "elapsed_ms": round(total, 1)}), flush=True)
    if any(row["error"] for row in rows):
        raise RuntimeError("one or more network trials failed; raw failure rows were retained")
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    audit = {"passed": len(rows) == args.trials and all(row["workflow_success"] for row in rows),
             "trial_rows": len(rows), "alternating_release_balance": {
                 tag: sum(row["tag"] == tag for row in rows) for tag, _ in RELEASES},
             "trials_sha256": sha256((args.output / "trials.jsonl").read_bytes()).hexdigest(),
             "protocol_sha256": sha256((args.output / "frozen_protocol.json").read_bytes()).hexdigest()}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
