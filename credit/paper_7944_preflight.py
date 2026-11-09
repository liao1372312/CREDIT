"""Check whether a machine and input manifest can run paper 7944 faithfully."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .paper_7944_protocol import PAPER_7944, validate_source_groups

REQUIRED_PACKAGES = {"numpy": "1.26.4", "scikit-learn": "1.6.1",
                     "torch": "2.6.0", "tiktoken": "0.9.0"}
TOOL_SANDBOX_FULL_SHA = "165848b9a78cead7ca7fe7c89c688b58e6501219"


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def inspect(root: Path, manifest: dict | None = None) -> dict:
    issues = []
    if sys.version_info[:3] != (3, 11, 9):
        issues.append("Python version differs from paper 3.11.9")
    for package, required in REQUIRED_PACKAGES.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        if installed != required:
            issues.append(f"{package}: expected {required}, found {installed or 'missing'}")
    tau = root / "vendor" / "tau-bench"
    frozen_tau = (root / "vendor" / "tau-bench-paper" /
                  f"tau-bench-{PAPER_7944.tau_bench_commit}")
    if (_git_head(tau) != PAPER_7944.tau_bench_commit and
            not (frozen_tau / "tau_bench").is_dir()):
        issues.append("tau-bench is not at the registered commit")
    sandbox = root / "vendor" / "ToolSandbox"
    frozen_archive = (root / "vendor" / "ToolSandbox-paper" /
                      f"ToolSandbox-{TOOL_SANDBOX_FULL_SHA}")
    if (_git_head(sandbox) != TOOL_SANDBOX_FULL_SHA and
            not (frozen_archive / "tool_sandbox").is_dir()):
        issues.append("ToolSandbox paper commit 165848b is unavailable")
    if manifest is None:
        issues.append("paper task IDs and source-group manifest are missing")
    else:
        try:
            validate_source_groups(manifest["splits"], manifest["groups_by_task"])
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"paper task manifest is invalid: {exc}")
    return {"ready": not issues, "issues": issues,
            "registered_protocol": "paper-7944", "paper_settings": {
                "actor": PAPER_7944.actor_model,
                "simulator": PAPER_7944.simulator_model,
                "embedding": PAPER_7944.embedding_model}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="JSON with splits and groups_by_task")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    report = inspect(root, manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
