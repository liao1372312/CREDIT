"""OpenAI-compatible JSON actor with exact-prompt caching and transport audit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter_ns, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from transformers import AutoTokenizer

from .llm_remote_policy_v2 import load_config
from .toolsandbox_local_v3 import MODEL


def _content(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item.get("text", "")) for item in value if isinstance(item, dict))
    raise ValueError("remote response has no textual message content")


class RemoteJSONAgent:
    """Small adapter exposing the same ``infer`` interface as QwenJSONAgent."""

    def __init__(self, output: Path, actor_seed: int, *, env: Path = Path(".env"),
                 tokenizer_path: Path = MODEL, temperature: float = 0.2,
                 max_new_tokens: int = 384, workers: int = 8,
                 exact_prompt_cache: bool = True):
        self.output = output
        self.config = load_config(env)
        self.actor_seed = int(actor_seed)
        self.temperature = float(temperature)
        self.max_new_tokens = min(int(max_new_tokens), self.config["max_tokens"])
        self.workers = int(workers)
        self.exact_prompt_cache = bool(exact_prompt_cache)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        self.cache: dict[str, dict] = {}
        self.calls = self.tokens_in = self.tokens_out = 0
        self.prompt_keys: set[str] = set()
        self.manifest = {
            "model": self.config["model"],
            "model_type": "remote OpenAI-compatible chat completions",
            "tokenizer_path": str(tokenizer_path.resolve()),
            "tokenizer_revision": tokenizer_path.name,
            "actor_seed": self.actor_seed,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "max_prompt_tokens": 12288,
            "parallel_connections": self.workers,
            "exact_prompt_cache": self.exact_prompt_cache,
            "base_url_sha256": self.config["base_url_sha256"],
            "credentials_persisted": False,
        }

    def _call(self, item: tuple[str, str, list[dict], int]) -> dict:
        key, rendered, chat, input_tokens = item
        request_value = {
            "model": self.config["model"], "messages": chat,
            "temperature": self.temperature, "seed": self.actor_seed,
            "max_tokens": self.max_new_tokens,
        }
        reasoning_effort = getattr(self, "reasoning_effort", None)
        if reasoning_effort is not None:
            request_value["reasoning_effort"] = reasoning_effort
        body = json.dumps(request_value, ensure_ascii=False, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        if self.config["api_key"]:
            headers["Authorization"] = "Bearer " + self.config["api_key"]
        last_error: Exception | None = None
        overall_began = perf_counter_ns()
        for attempt in range(1, 4):
            began = perf_counter_ns()
            try:
                request = Request(self.config["url"], body, headers, method="POST")
                with urlopen(request, timeout=self.config["timeout"]) as response:
                    raw = response.read()
                successful_attempt_latency_ms = (perf_counter_ns() - began) / 1e6
                latency_ms = (perf_counter_ns() - overall_began) / 1e6
                payload = json.loads(raw)
                choice = payload["choices"][0]
                message = choice.get("message") or {}
                try:
                    text = _content(message.get("content"))
                except ValueError as error:
                    safe_shape = {
                        "message_fields": sorted(message.keys()) if isinstance(message, dict) else [],
                        "content_type": type(message.get("content")).__name__ if isinstance(message, dict) else type(message).__name__,
                        "reasoning_content_present": bool(message.get("reasoning_content")) if isinstance(message, dict) else False,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": payload.get("usage"),
                    }
                    raise ValueError("remote response has no textual message content; " +
                                     json.dumps(safe_shape, sort_keys=True)) from error
                usage = payload.get("usage") or {}
                output_tokens = int(usage.get("completion_tokens") or
                                    len(self.tokenizer.encode(text, add_special_tokens=False)))
                fingerprint = payload.get("system_fingerprint")
                return {
                    "prompt_sha256": key,
                    "messages_rendered": rendered,
                    "agent_output": text,
                    "rendered_input_tokens": input_tokens,
                    "input_tokens": int(usage.get("prompt_tokens") or input_tokens),
                    "output_tokens": output_tokens,
                    "hit_generation_limit": (choice.get("finish_reason") == "length"),
                    "remote_latency_ms": latency_ms,
                    "successful_attempt_latency_ms": successful_attempt_latency_ms,
                    "network_attempts": attempt,
                    "provider_model": payload.get("model") or self.config["model"],
                    "system_fingerprint_sha256": (sha256(str(fingerprint).encode()).hexdigest()
                                                   if fingerprint is not None else None),
                    "request_sha256": sha256(body).hexdigest(),
                    "response_sha256": sha256(raw).hexdigest(),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": usage,
                }
            except HTTPError as error:
                last_error = RuntimeError("remote actor HTTP status " + str(error.code))
                if error.code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except (URLError, TimeoutError) as error:
                last_error = RuntimeError("remote actor connection failed: " + type(error).__name__)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                detail = str(error).replace("\n", " ")[:120]
                last_error = RuntimeError("remote actor response parse failed: " +
                                          type(error).__name__ + ": " + detail)
            if attempt < 3:
                sleep(2 ** (attempt - 1))
        assert last_error is not None
        raise last_error

    def infer(self, chats: list[list[dict]]) -> list[dict]:
        rendered = [self.tokenizer.apply_chat_template(chat, tokenize=False,
                    add_generation_prompt=True) for chat in chats]
        keys = [sha256(text.encode()).hexdigest() for text in rendered]
        if not self.exact_prompt_cache:
            pending = []
            for key, text, chat in zip(keys, rendered, chats):
                n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
                if n_tokens > 12288:
                    raise ValueError("prompt exceeds frozen budget; no silent truncation")
                pending.append((key, text, chat, n_tokens))
            before = perf_counter_ns()
            records, failures = [], []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self._call, item) for item in pending]
                for future in futures:
                    try:
                        records.append(future.result())
                    except Exception as error:
                        failures.append(error)
            if records:
                with (self.output / "generations.jsonl").open("a") as stream:
                    for record in records:
                        record["generation_uid"] = f"actor-{self.actor_seed}-generation-{self.calls:06d}"
                        self.calls += 1
                        self.prompt_keys.add(record["prompt_sha256"])
                        self.tokens_in += record["input_tokens"]
                        self.tokens_out += record["output_tokens"]
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
            print(json.dumps({"actor_seed": self.actor_seed, "live_requests": self.calls,
                "new_requests": len(records),
                "failed_requests": len(failures),
                "batch_seconds": round((perf_counter_ns() - before) / 1e9, 2)}), flush=True)
            if failures:
                raise failures[0]
            return [deepcopy(record) for record in records]
        pending = []
        seen = set()
        for key, text, chat in zip(keys, rendered, chats):
            if key in self.cache or key in seen:
                continue
            seen.add(key)
            n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            if n_tokens > 12288:
                raise ValueError("prompt exceeds frozen budget; no silent truncation")
            pending.append((key, text, chat, n_tokens))
        if pending:
            before = perf_counter_ns()
            records, failures = [], []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self._call, item) for item in pending]
                for future in futures:
                    try:
                        records.append(future.result())
                    except Exception as error:
                        failures.append(error)
            if records:
                with (self.output / "generations.jsonl").open("a") as stream:
                    for record in records:
                        record["generation_uid"] = f"actor-{self.actor_seed}-generation-{self.calls:06d}"
                        self.cache[record["prompt_sha256"]] = record
                        self.calls += 1
                        self.prompt_keys.add(record["prompt_sha256"])
                        self.tokens_in += record["input_tokens"]
                        self.tokens_out += record["output_tokens"]
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            print(json.dumps({"actor_seed": self.actor_seed,
                "generated_contexts": self.calls, "new_contexts": len(records),
                "failed_contexts": len(failures),
                "batch_seconds": round((perf_counter_ns() - before) / 1e9, 2)}), flush=True)
            if failures:
                raise failures[0]
        return [deepcopy(self.cache[key]) for key in keys]

    def close(self) -> None:
        self.manifest.update({
            "generation_records": self.calls,
            "unique_generation_contexts": len(self.prompt_keys),
            "input_tokens": self.tokens_in,
            "output_tokens": self.tokens_out,
            "cache": ("exact rendered-prompt cache within actor seed" if self.exact_prompt_cache
                      else "disabled; every actor turn made a live provider request"),
            "live_api_calls": self.calls,
        })
        (self.output / "agent_manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2))
