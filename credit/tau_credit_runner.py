"""CREDIT adapter for the pinned original tau-bench simulator.

The upstream task, policy, tool, state, and evaluator implementations are used
unchanged. Actor and user messages go through the repository's OpenAI-compatible
endpoint configuration; logs retain hashes and usage, not conversation values.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .llm_remote_policy_v2 import load_config

ROOT = Path(__file__).resolve().parents[1]
TAU_SOURCE = ROOT / "vendor/tau-bench"
POOL_ROOT = ROOT / "datasets/tau_original_credit_pool_v2"
MAX_ACTOR_STEPS = 30
LOCAL_QWEN = Path(
    "/home/lxc/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class RemoteChat:
    def __init__(self, env_path: Path, seed: int, *, temperature: float = 0.0,
                 max_tokens: int = 384):
        self.config = load_config(env_path)
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.max_tokens = min(int(max_tokens), self.config["max_tokens"])
        self.records: list[dict] = []

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        request_value = {
            "model": self.config["model"], "messages": messages,
            "temperature": self.temperature, "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        if tools:
            request_value["tools"] = tools
            request_value["tool_choice"] = "auto"
        body = json.dumps(request_value, ensure_ascii=False, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        if self.config["api_key"]:
            headers["Authorization"] = "Bearer " + self.config["api_key"]
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                req = Request(self.config["url"], body, headers, method="POST")
                began = perf_counter()
                with urlopen(req, timeout=self.config["timeout"]) as response:
                    raw = response.read()
                latency = perf_counter() - began
                payload = json.loads(raw)
                choice = payload["choices"][0]
                message = choice["message"]
                result = {
                    "message": message, "finish_reason": choice.get("finish_reason"),
                    "usage": payload.get("usage") or {},
                    "provider_model": payload.get("model") or self.config["model"],
                    "response_sha256": digest(raw), "request_sha256": digest(body),
                    "prompt_sha256": digest(json.dumps(messages, ensure_ascii=False,
                                                        sort_keys=True).encode()),
                    "latency_seconds": latency, "network_attempts": attempt,
                    "system_fingerprint_sha256": digest(str(payload.get("system_fingerprint")).encode())
                        if payload.get("system_fingerprint") is not None else None,
                }
                self.records.append({k: v for k, v in result.items() if k != "message"})
                return result
            except HTTPError as exc:
                # Keep a short provider diagnostic for reproducibility while
                # removing any echoed credential before it reaches a log.
                raw_error = exc.read(2048).decode("utf-8", "replace")
                if self.config["api_key"]:
                    raw_error = raw_error.replace(self.config["api_key"], "[REDACTED]")
                try:
                    error_payload = json.loads(raw_error)
                    provider_message = ((error_payload.get("error") or {}).get("message")
                                        or error_payload.get("message") or "unspecified")
                except (TypeError, ValueError):
                    provider_message = "unparseable"
                provider_message = str(provider_message).replace("\n", " ")[:240]
                last = RuntimeError(f"provider HTTP {exc.code}: {provider_message}")
                if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except (URLError, TimeoutError) as exc:
                last = RuntimeError(f"provider connection failed: {type(exc).__name__}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last = RuntimeError(f"provider response parse failed: {type(exc).__name__}")
                break
        assert last is not None
        raise last


class LocalQwenChat:
    """Local Qwen2.5 chat backend; actor and simulator share one checkpoint."""
    _engine = None

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.records: list[dict] = []
        if LocalQwenChat._engine is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not LOCAL_QWEN.exists():
                raise FileNotFoundError(f"local Qwen checkpoint not found: {LOCAL_QWEN}")
            tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_QWEN), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(LOCAL_QWEN), local_files_only=True, torch_dtype=torch.bfloat16,
                device_map="auto", low_cpu_mem_usage=True,
            )
            model.eval()
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            LocalQwenChat._engine = (torch, tokenizer, model)
        self.torch, self.tokenizer, self.model = LocalQwenChat._engine

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        rendered_messages = deepcopy(messages)
        for entry in rendered_messages:
            for call in entry.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
        prompt = self.tokenizer.apply_chat_template(
            rendered_messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = self.model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        began = perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded, max_new_tokens=512, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = perf_counter() - began
        output_ids = generated[0, encoded["input_ids"].shape[1]:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=False).strip()
        calls = []
        for index, match in enumerate(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)):
            try:
                call = json.loads(match.group(1))
                calls.append({
                    "id": f"local-call-{index}", "type": "function",
                    "function": {"name": call["name"],
                                 "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False)},
                })
            except (KeyError, TypeError, ValueError):
                continue
        content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip() or None
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        request_raw = json.dumps({"messages": rendered_messages, "tools": tools},
                                 ensure_ascii=False, sort_keys=True).encode()
        response_raw = text.encode()
        usage = {"prompt_tokens": int(encoded["input_ids"].shape[1]),
                 "completion_tokens": int(output_ids.shape[0])}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        self.records.append({
            "request_sha256": digest(request_raw), "response_sha256": digest(response_raw),
            "prompt_sha256": digest(prompt.encode()), "usage": usage,
            "provider_model": "Qwen2.5-7B-Instruct (local)", "latency_seconds": elapsed,
            "network_attempts": 0,
        })
        return {"message": message, "usage": usage,
                "provider_model": "Qwen2.5-7B-Instruct (local)",
                "response_sha256": digest(response_raw)}


def make_chat(backend: str, env_path: Path, seed: int):
    if backend == "local-qwen":
        return LocalQwenChat(seed)
    if backend == "remote":
        return RemoteChat(env_path, seed)
    raise ValueError(f"unsupported backend: {backend}")


class TauRemoteUser:
    """OpenAI-compatible implementation of the original benchmark LLM user."""
    def __init__(self, client: RemoteChat):
        self.client = client
        self.messages: list[dict] = []

    def reset(self, instruction: str | None = None) -> str:
        instruction_text = (("\n\nInstruction: " + instruction + "\n")
                            if instruction is not None else "")
        system = f"""You are a user interacting with an agent.{instruction_text}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": "Hi! How can I help you today?"}]
        return self._next()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self._next()

    def _next(self) -> str:
        result = self.client.complete(self.messages)
        message = result["message"]
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("user simulator returned no text")
        self.messages.append({"role": "assistant", "content": content})
        return "###STOP###" if "###STOP###" in content else content

    def get_total_cost(self) -> float:
        # Provider-specific price metadata is not reliably returned by this endpoint.
        return 0.0


def install_tau_imports():
    import sys
    sys.path.insert(0, str(TAU_SOURCE.resolve()))
    import tau_bench.envs.base as tau_base
    from tau_bench.envs import get_env
    from tau_bench.types import Action, RESPOND_ACTION_NAME, SolveResult

    active_user_client: dict[str, RemoteChat] = {}

    def load_remote_user(user_strategy, model, provider):
        # Every benchmark user is the pinned original LLM user strategy; model
        # and provider identity are frozen in the run manifest.
        client = active_user_client["client"]
        return TauRemoteUser(client)

    tau_base.load_user = load_remote_user
    return get_env, Action, RESPOND_ACTION_NAME, SolveResult, active_user_client


def run_episode(workload: str, task_id: int, seed: int, memory: list[dict],
                *, env_path: Path, backend: str = "remote",
                user_model_label: str = "configured-endpoint") -> dict:
    if len(memory) > 2:
        raise ValueError("public CREDIT protocol exposes at most two memory records")
    if any(not isinstance(item.get("text"), str) for item in memory):
        raise ValueError("every exposed record must contain memory text")
    get_env, Action, respond_name, SolveResult, active_user = install_tau_imports()
    random.seed(seed)
    actor = make_chat(backend, env_path, seed)
    user = make_chat(backend, env_path, seed + 1)
    active_user["client"] = user
    env = get_env(workload, user_strategy="llm", user_model=user_model_label,
                  user_provider="openai", task_split="test", task_index=task_id)
    reset = env.reset(task_index=task_id)
    env.wiki = env.wiki + "\n\nHistorical experience (advisory; do not copy historical values):\n" + \
        json.dumps([item["text"] for item in memory], ensure_ascii=False)
    messages = [{"role": "system", "content": env.wiki},
                {"role": "user", "content": reset.observation}]
    reward = 0.0
    info = reset.info.model_dump()
    action_count = 0
    started = perf_counter()
    for _ in range(MAX_ACTOR_STEPS):
        response = actor.complete(messages, env.tools_info)
        message = response["message"]
        calls = message.get("tool_calls") or []
        if calls:
            call = calls[0]
            fn = call.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("model tool arguments are not a JSON object")
            action = Action(name=fn["name"], kwargs=args)
            step = env.step(action)
            action_count += 1
            reward = float(step.reward)
            info = {**info, **step.info.model_dump()}
            # Match the original tau-bench tool-calling agent: discard any
            # additional calls from the same generation and execute only #1.
            assistant_message = dict(message)
            assistant_message["tool_calls"] = [call]
            messages.extend([
                assistant_message,
                {"role": "tool", "tool_call_id": call.get("id") or "tau-call",
                 "name": fn["name"], "content": step.observation},
            ])
            if step.done:
                break
            continue
        else:
            action = Action(name=respond_name, kwargs={"content": message.get("content") or ""})
            messages.append(message)
        step = env.step(action)
        reward = float(step.reward)
        info = {**info, **step.info.model_dump()}
        # The raw interaction can include user identifiers. It stays in memory
        # only for the live model context and is never written to the artifact.
        messages.append({"role": "user", "content": step.observation})
        if step.done:
            break
    task = env.task
    return {
        "workload": workload, "task_id": task_id, "seed": seed,
        "task_family": f"{workload}:task-{task_id}",
        "task_instruction_sha256": digest(task.instruction.encode()),
        "exposure_uids": [item["uid"] for item in memory],
        "reward": reward, "success": bool(reward >= 1.0 - 1e-9),
        "actor_generations": len(actor.records),
        "user_simulator_generations": len(user.records),
        "actor_and_user_generations": len(actor.records) + len(user.records),
        "tool_calls": action_count,
        "prompt_tokens": sum(int((r.get("usage") or {}).get("prompt_tokens") or 0)
                              for r in actor.records + user.records),
        "completion_tokens": sum(int((r.get("usage") or {}).get("completion_tokens") or 0)
                                  for r in actor.records + user.records),
        "http_attempts": sum(int(r.get("network_attempts", 1)) for r in actor.records + user.records),
        "response_hashes": [r["response_sha256"] for r in actor.records + user.records],
        "request_hashes": [r["request_sha256"] for r in actor.records + user.records],
        "provider_models": sorted({r["provider_model"] for r in actor.records + user.records}),
        "system_fingerprint_hashes": sorted({r["system_fingerprint_sha256"] for r in actor.records + user.records
                                               if r.get("system_fingerprint_sha256")}),
        "duration_seconds": perf_counter() - started,
        "executor": "pinned original tau-bench local tool/state/evaluator",
        "backend": backend,
        # The upstream evaluator info embeds full task instructions and output
        # strings. Save only a digest and non-sensitive reward diagnostics.
        "evaluator_info_sha256": digest(json.dumps(info, ensure_ascii=False,
                                                      sort_keys=True, default=str).encode()),
        "evaluator_diagnostics": {
            "source": info.get("source"),
            "reward_type": (info.get("reward_info") or {}).get("type")
                if isinstance(info.get("reward_info"), dict) else None,
            "r_actions": (info.get("reward_info") or {}).get("r_actions")
                if isinstance(info.get("reward_info"), dict) else None,
            "r_outputs": (info.get("reward_info") or {}).get("r_outputs")
                if isinstance(info.get("reward_info"), dict) else None,
        },
    }


def load_pool(domain: str) -> tuple[list[dict], dict]:
    manifest_path = POOL_ROOT / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pool_path = POOL_ROOT / f"{domain}_memory_pool_64.json"
    pool = json.loads(pool_path.read_text())
    if len(pool) != 64 or any(r["split"] != "development" for r in pool):
        raise ValueError("CREDIT source pool is not 64 development-only records")
    if len({r["abstract_chain_sha256"] for r in pool}) != len(pool):
        raise ValueError("duplicate workflow chain in source pool")
    return pool, manifest


def write_smoke(workload: str, output: Path, env_path: Path, seed: int, task_id: int,
                backend: str) -> None:
    pool, protocol = load_pool(workload)
    row = run_episode(workload, task_id, seed, [], env_path=env_path, backend=backend)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "smoke only; one no-memory episode; not a CREDIT performance result",
        "source_pool_manifest_sha256": digest((POOL_ROOT / "split_manifest.json").read_bytes()),
        "source_records_available": len(pool),
        "tau_source_commit": protocol["source_commit"],
        "episode": row,
    }
    (output / "smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"smoke": "completed", "workload": workload,
                      "task_id": task_id, "reward": row["reward"],
                      "generations": row["actor_and_user_generations"],
                      "tool_calls": row["tool_calls"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("retail", "airline"), required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--backend", choices=("remote", "local-qwen"), default="remote")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_smoke(args.workload, args.output, args.env, args.seed, args.task_id, args.backend)


if __name__ == "__main__":
    main()
