"""Fetch the two public benchmark source archives at paper 7944's commits."""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from .paper_7944_preflight import TOOL_SANDBOX_FULL_SHA
from .paper_7944_protocol import PAPER_7944

BENCHMARKS = {
    "ToolSandbox": ("apple-aiml-research/ToolSandbox", TOOL_SANDBOX_FULL_SHA,
                    "ToolSandbox-paper", "ToolSandbox", "tool_sandbox"),
    "tau-bench": ("sierra-research/tau-bench", PAPER_7944.tau_bench_commit,
                  "tau-bench-paper", "tau-bench", "tau_bench"),
}


def fetch(name: str, root: Path) -> dict:
    if name not in BENCHMARKS:
        raise ValueError("unknown paper benchmark")
    repository, commit, directory, prefix, package = BENCHMARKS[name]
    vendor = (root / "vendor").resolve()
    destination = (vendor / directory / f"{prefix}-{commit}").resolve()
    if not destination.is_relative_to(vendor):
        raise ValueError("benchmark target escapes the vendor directory")
    if destination.exists():
        if not (destination / package).is_dir():
            raise ValueError("existing benchmark target is incomplete")
        return {"name": name, "commit": commit, "path": str(destination),
                "already_present": True}
    url = f"https://codeload.github.com/{repository}/zip/{commit}"
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    archive_digest = sha256(payload).hexdigest()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(payload)) as archive:
        members = archive.infolist()
        expected_prefix = f"{prefix}-{commit}/"
        if not members or any(
                not member.filename.startswith(expected_prefix) or
                Path(member.filename).is_absolute() or
                ".." in Path(member.filename).parts
                for member in members):
            raise ValueError("benchmark archive contains an unexpected path")
        archive.extractall(parent)
    if not (destination / package).is_dir():
        raise ValueError("benchmark archive lacks the expected Python package")
    (destination / ".paper_archive_sha256").write_text(archive_digest + "\n",
                                                         encoding="ascii")
    return {"name": name, "commit": commit, "path": str(destination),
            "archive_sha256": archive_digest, "already_present": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=tuple(BENCHMARKS),
                        help="default: both paper benchmarks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name in args.names or BENCHMARKS:
        print(fetch(name, root))


if __name__ == "__main__":
    main()
