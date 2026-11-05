"""Post-hoc transitive source-closure audit for selector attribution.

This audit does not rewrite the frozen protocol.  It records which local
dependencies were frozen by the run, which are witnessed by an earlier frozen
bundle with the same bytes, and which can only be snapshotted after execution.
"""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
CREDIT = ROOT / "credit"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def local_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        candidates = ([node.module.split(".")[0]] if node.module else
                      [alias.name.split(".")[0] for alias in node.names])
        imports.update(name for name in candidates
                       if (CREDIT / f"{name}.py").exists())
    return imports


def closure(starts: set[str]) -> set[str]:
    found: set[str] = set()
    pending = list(starts)
    while pending:
        name = pending.pop()
        path = CREDIT / f"{name}.py"
        if name in found or not path.exists():
            continue
        found.add(name)
        pending.extend(local_imports(path) - found)
    return found


def prior_witnesses(name: str, expected: str,
                    excluded_roots: tuple[Path, ...]) -> list[str]:
    witnesses = []
    for candidate in (ROOT / "results").glob(f"**/source/{name}.py"):
        for excluded in excluded_roots:
            try:
                candidate.resolve().relative_to(excluded.resolve())
                break
            except ValueError:
                continue
        else:
            if digest(candidate) == expected:
                witnesses.append(str(candidate.relative_to(ROOT)))
            continue
    return sorted(witnesses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    snapshot = args.output / "source"
    snapshot.mkdir()

    protocol_path = args.results / "frozen_protocol.json"
    execution_audit_path = args.results / "execution_audit.json"
    protocol = json.loads(protocol_path.read_text())
    execution_audit = json.loads(execution_audit_path.read_text())
    if not execution_audit.get("verified"):
        raise ValueError("execution audit has not passed")

    starts = {Path(name).stem for name in protocol["source_sha256"]}
    starts.add("toolsandbox_m_selector_attribution_verify")
    modules = closure(starts)
    frozen = protocol["source_sha256"]
    records = []
    for name in sorted(modules):
        filename = f"{name}.py"
        path = CREDIT / filename
        current_hash = digest(path)
        shutil.copy2(path, snapshot / filename)
        witnesses = prior_witnesses(
            name, current_hash, (args.results, args.output))
        if filename in frozen:
            if frozen[filename] != current_hash:
                raise ValueError("current source differs from frozen run: " + filename)
            status = "pre_frozen_current_run"
        elif witnesses:
            status = "matched_prior_frozen_bundle"
        else:
            status = "posthoc_only"
        records.append({"module": name, "sha256": current_hash,
            "status": status, "imports": sorted(local_imports(path)),
            "prior_frozen_witnesses": witnesses})

    missing = [row for row in records
               if row["status"] != "pre_frozen_current_run"]
    posthoc_only = [row["module"] for row in records
                    if row["status"] == "posthoc_only"]
    report = {
        "audit_type": "posthoc_transitive_dependency_closure",
        "recorded_after_remote_execution": True,
        "does_not_upgrade_missing_files_to_prefrozen": True,
        "execution_audit_verified": True,
        "protocol_sha256": digest(protocol_path),
        "execution_audit_sha256": digest(execution_audit_path),
        "python": sys.version,
        "entry_modules": sorted(starts),
        "local_modules": len(records),
        "pre_frozen_current_run": sum(
            row["status"] == "pre_frozen_current_run" for row in records),
        "missing_from_original_freeze": len(missing),
        "posthoc_only_modules": posthoc_only,
        "dependencies": records,
        "interpretation": (
            "Exact replay remains valid for the current checkout. Matching prior "
            "bundles provide byte-identical historical witnesses, but this post-hoc "
            "manifest is not evidence that every dependency was registered by the "
            "selector-attribution protocol before remote calls."
        ),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "modules": len(records),
        "missing_from_original_freeze": len(missing),
        "posthoc_only_modules": posthoc_only,
        "report_sha256": digest(report_path)}, indent=2))


if __name__ == "__main__":
    main()
