"""Offline constrained Qwen decision policy for finite instrumented contexts.

Each context is scored once by the real model; deterministic prompt scores
are cached. Subsequent constrained action samples reuse exactly that policy.
The mechanism learner receives sampled actions, not these probabilities.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .stateful import ACTIONS, MEMORIES, NEIGHBORS, ROLES

MEMORY_TEXT = {
    "lookup_key": "Use the standard lookup request with the current record identifier.",
    "transform_rule": "When required inputs exist, use the standard transform request.",
    "submit_field": "After authorization, submit using the contract mode stated in the notes.",
    "legacy_contract": "The saved submit contract accepts standard mode; patched mode is unsupported.",
    "new_contract": "The saved submit contract accepts patched mode; standard mode is unsupported.",
    "auth_pair_a": "Credential note A: with both credential notes A and B, prefer standard authorization.",
    "auth_pair_b": "Credential note B: with both credential notes A and B, prefer standard authorization.",
    "repair_rule": "If prerequisites are missing, use repair mode to restore required inputs before continuing.",
    "conflict": "An older operational note says to send an invalid request for this operation.",
    "noop": "Unrelated note about the color of a website banner.",
    "cache_hint": "The cached record identifier is valid for a standard lookup.",
    "format_hint": "The standard transform request uses the supported field format.",
    "retry_hint": "Use repair authorization to resolve incomplete setup or retry a transient failure.",
    "quota_hint": "Standard authorization is allowed under the recorded quota.",
    "obsolete_lookup": "An older lookup note recommends patched mode instead of standard mode.",
    "hidden_fix": "The submit payload correction applies to either standard or patched contract mode.",
}

SYSTEM = """You are a tool agent choosing a request mode for the next scheduled operation.
Choose one letter only: A standard, B patched, C repair, D invalid.
Standard performs the normal operation and needs its prerequisites.
Patched uses an alternate contract and is useful only if that contract is accepted.
Repair can restore prerequisites; authorization repair still needs a record.
Invalid sends a malformed request. Saved notes may be obsolete or conflicting.
Choose the request most likely to succeed using the observable context and notes.
Do not infer a provider change unless the notes or responses give evidence."""


def contexts():
    for role in ROLES:
        neighbors = NEIGHBORS[role]
        for mask in range(1 << len(neighbors)):
            exposure = [memory for i, memory in enumerate(neighbors) if mask & (1 << i)]
            for missing in ((0,) if role == "lookup" else (0, 1)):
                for transient in (0, 1):
                    notes = "\n".join(f"- {memory}: {MEMORY_TEXT[memory]}" for memory in exposure) or "No saved notes."
                    prompt = (f"Operation: {role}\nRequired prerequisites: {'missing' if missing else 'present'}\n"
                              f"Last response signal: {'transient failure' if transient else 'no transient failure'}\n"
                              f"Retrieved local memory notes:\n{notes}\nChoose A, B, C, or D.")
                    yield (role, missing, transient, mask), prompt


def score_model(model_path: Path, output: Path, *, batch_size=8):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    choices = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
    if any(len(ids) != 1 for ids in choices):
        raise ValueError("choice letters must each map to a single token")
    token_ids = torch.tensor([ids[0] for ids in choices], device="cuda")
    data = list(contexts())
    rows, input_tokens = [], 0
    with torch.inference_mode():
        for offset in range(0, len(data), batch_size):
            batch = data[offset:offset + batch_size]
            chats = [tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True) for _, prompt in batch]
            encoded = tokenizer(chats, return_tensors="pt", padding=True).to("cuda")
            input_tokens += int(encoded["attention_mask"].sum())
            scores = model(**encoded).logits[:, -1, token_ids].float()
            probabilities = scores.softmax(dim=1).cpu().numpy()
            logits = scores.cpu().numpy()
            for (key, prompt), rendered, probability, score in zip(batch, chats, probabilities, logits):
                rows.append({"key": list(key), "probabilities": probability.tolist(), "choice_logits": score.tolist(),
                             "prompt": prompt, "rendered_prompt_sha256": sha256(rendered.encode()).hexdigest()})
            print(json.dumps({"contexts_scored": len(rows), "total": len(data)}), flush=True)
    payload = json.dumps(rows, indent=2).encode()
    (output / "decision_scores.json").write_bytes(payload)
    manifest = {
        "model_path": str(model_path.resolve()), "model_revision": model_path.name,
        "model_type": model.config.model_type, "policy": "A/B/C/D next-token constrained softmax, temperature 1",
        "choice_token_ids": token_ids.tolist(), "actions": ACTIONS, "system_prompt": SYSTEM,
        "memory_text": MEMORY_TEXT, "llm_context_forward_evaluations": len(rows),
        "forward_batches": (len(rows) + batch_size - 1) // batch_size, "input_tokens": input_tokens,
        "elapsed_seconds": perf_counter() - start, "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        "decision_scores_sha256": sha256(payload).hexdigest(),
        "adapter_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_files": {path.name: {"bytes": path.stat().st_size,
                                    "sha256": sha256(path.read_bytes()).hexdigest()}
                        for path in sorted(model_path.iterdir()) if path.is_file()},
        "limits": ["fixed workflow and constrained request modes", "synthetic API snapshots",
                   "cached identical finite-context prompts, not fresh end-to-end multi-turn generation",
                   "score probabilities stay in sandbox; learners see independent sampled action counts"],
    }
    (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def install_policy(scores_path: Path):
    """Install the frozen environment policy, leaving the learner's inputs unchanged."""
    import credit.stateful as environment
    rows = json.loads(scores_path.read_text())
    scores = {tuple(row["key"]): np.asarray(row["probabilities"], dtype=float) for row in rows}
    # Float32 softmax output is renormalized, then pinned by the saved file hash.
    scores = {key: probability / probability.sum() for key, probability in scores.items()}
    def policy(role, missing, transient, exposure):
        return scores[(role, missing, transient, environment.local_mask(role, exposure))]
    environment._policy = policy
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/llm_policy"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(score_model(args.model, args.output, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
