"""Persistent replay sentinel for temporal drift of a closed LLM provider."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter_ns, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .llm_policy import SYSTEM, contexts
from .llm_remote_policy_v2 import answer_letter, load_config
from .user_data import digest


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def wilson(changed: int, total: int) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = changed / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0.0, center - half), min(1.0, center + half)]


def request_one(config: dict, index: int, key: tuple, prompt: str) -> dict:
    body = json.dumps({"model": config["model"],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": config["max_tokens"]}, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = "Bearer " + config["api_key"]
    last_error = None
    overall_began = perf_counter_ns()
    for attempt in range(1, 4):
        began = perf_counter_ns()
        try:
            with urlopen(Request(config["url"], body, headers, method="POST"),
                         timeout=config["timeout"]) as response:
                raw = response.read()
            successful_attempt_latency = (perf_counter_ns() - began) / 1e6
            latency = (perf_counter_ns() - overall_began) / 1e6
            payload = json.loads(raw)
            first = payload["choices"][0]
            reply = first.get("message", {}).get("content")
            letter = answer_letter(reply)
            fingerprint = payload.get("system_fingerprint")
            return {"index": index, "key": list(key), "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                "letter": letter, "reply": reply, "network_latency_ms": latency,
                "successful_attempt_latency_ms": successful_attempt_latency,
                "network_attempts": attempt, "provider_model": payload.get("model") or config["model"],
                "system_fingerprint_sha256": (sha256(str(fingerprint).encode()).hexdigest()
                                               if fingerprint is not None else None),
                "request_sha256": sha256(body).hexdigest(), "response_sha256": sha256(raw).hexdigest(),
                "response_id_sha256": (sha256(str(payload.get("id")).encode()).hexdigest()
                                        if payload.get("id") else None),
                "finish_reason": first.get("finish_reason"), "usage": payload.get("usage") or {}}
        except HTTPError as error:
            last_error = RuntimeError("sentinel HTTP status " + str(error.code))
            if error.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError) as error:
            last_error = RuntimeError("sentinel connection failed: " + type(error).__name__)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            last_error = RuntimeError("sentinel parse failed: " + type(error).__name__)
            break
        if attempt < 3:
            sleep(2 ** (attempt - 1))
    raise last_error or RuntimeError("unknown sentinel failure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--baseline", type=Path,
        default=Path("results/credit_v3_remote_llm_policy_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/provider_drift_monitor_v1"))
    parser.add_argument("--sentinels", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.env)
    entries = list(contexts())
    baseline_rows = read_jsonl(args.baseline / "responses.jsonl")
    baseline_protocol = json.loads((args.baseline / "frozen_protocol.json").read_text())
    if len(entries) != len(baseline_rows):
        raise ValueError("baseline response inventory is incomplete")
    if args.sentinels < 2 or args.sentinels > len(entries):
        raise ValueError("sentinel count outside inventory")
    indices = sorted({round(i * (len(entries) - 1) / (args.sentinels - 1))
                      for i in range(args.sentinels)})
    if len(indices) != args.sentinels:
        raise ValueError("sentinel spacing produced duplicates")
    for index in indices:
        key, prompt = entries[index]
        old = baseline_rows[index]
        if old["key"] != list(key) or old["prompt_sha256"] != sha256(prompt.encode()).hexdigest():
            raise ValueError("historical baseline does not match current sentinel prompt")
    baseline_mtime = datetime.fromtimestamp((args.baseline / "responses.jsonl").stat().st_mtime,
                                            tz=timezone.utc)
    monitor = {"suite": "closed-provider temporal replay sentinel",
        "model_alias": config["model"], "base_url_sha256": config["base_url_sha256"],
        "temperature": 0, "max_output_tokens": config["max_tokens"],
        "inventory": len(entries), "sentinel_indices": indices,
        "sentinel_prompt_sha256": [sha256(entries[i][1].encode()).hexdigest() for i in indices],
        "historical_baseline": str(args.baseline.resolve()),
        "historical_baseline_sha256": digest(args.baseline / "responses.jsonl"),
        "historical_artifact_mtime_utc": baseline_mtime.isoformat(),
        "historical_collection_time_limit": "baseline did not persist an explicit collection timestamp; filesystem mtime is used",
        "request_equivalence": "same system, user prompt, model alias, temperature, and output cap; no seed in either request",
        "credentials_persisted": False, "source_sha256": digest(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=True)
    monitor_path = args.output / "monitor_protocol.json"
    if monitor_path.exists():
        previous = json.loads(monitor_path.read_text())
        immutable = {key: value for key, value in monitor.items() if key != "source_sha256"}
        old_immutable = {key: value for key, value in previous.items() if key != "source_sha256"}
        if immutable != old_immutable:
            raise ValueError("monitor protocol changed; use a new output directory")
    else:
        monitor_path.write_text(json.dumps(monitor, indent=2))
    collected = datetime.now(timezone.utc)
    wave_id = collected.strftime("%Y%m%dT%H%M%S.%fZ")
    waves = args.output / "waves"; waves.mkdir(exist_ok=True)
    selected = [(index, *entries[index]) for index in indices]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        fresh = list(executor.map(lambda item: request_one(config, *item), selected))
    for row in fresh:
        row["historical_letter"] = baseline_rows[row["index"]]["letter"]
        row["decision_changed"] = row["letter"] != row["historical_letter"]
    wave_path = waves / f"{wave_id}.jsonl"
    with wave_path.open("w") as stream:
        for row in fresh:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    changed = sum(row["decision_changed"] for row in fresh)
    span_hours = (collected - baseline_mtime).total_seconds() / 3600
    by_operation = {}
    for operation in sorted({row["key"][0] for row in fresh}):
        subset = [row for row in fresh if row["key"][0] == operation]
        by_operation[operation] = {"sentinels": len(subset),
            "decision_change_rate": sum(row["decision_changed"] for row in subset) / len(subset)}
    summary = {"wave_id": wave_id, "collection_date_utc": collected.isoformat(),
        "historical_artifact_mtime_utc": baseline_mtime.isoformat(),
        "observed_span_hours": span_hours, "sentinels": len(fresh),
        "decision_changes": changed, "decision_change_rate": changed / len(fresh),
        "decision_change_rate_wilson_CI95": wilson(changed, len(fresh)),
        "by_operation": by_operation,
        "provider_models": sorted({row["provider_model"] for row in fresh}),
        "system_fingerprints": sorted({row["system_fingerprint_sha256"] for row in fresh
                                       if row["system_fingerprint_sha256"]}),
        "mean_network_latency_ms": sum(row["network_latency_ms"] for row in fresh) / len(fresh),
        "long_term_evidence_available": span_hours >= 24 * 30,
        "interpretation_limit": "The monitor is operational; the observed span is not long-term until repeated waves cover at least 30 days.",
        "wave_sha256": digest(wave_path)}
    (waves / f"{wave_id}.summary.json").write_text(json.dumps(summary, indent=2))
    index_path = args.output / "wave_index.jsonl"
    with index_path.open("a") as stream:
        stream.write(json.dumps(summary) + "\n"); stream.flush(); os.fsync(stream.fileno())
    audit = {"passed": len(fresh) == args.sentinels and all(row["network_latency_ms"] > 0 for row in fresh),
        "prompt_hashes_match_historical_baseline": True, "request_failures": 0,
        "credentials_persisted": False, "wave_rows": len(fresh), "wave_sha256": digest(wave_path)}
    (waves / f"{wave_id}.audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
