"""Replay documented upstream Web API changes in isolated release runtimes.

The suite uses official json-server npm releases and official go-httpbin OCI
images.  It records upstream tag/package identities, restored database hashes,
raw response hashes, and a second replay audit.  No change is injected by this
repository.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from hashlib import sha256, sha512
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


JSON_SERVER = {
    "v0.17.4": {
        "commit": "10647dcf8aeea5d410464fcda6621191a7305e8c",
        "integrity": "sha512-bGBb0WtFuAKbgI7JV3A864irWnMZSvBYRJbohaOuatHwKSRFUfqtQlrYMrB6WbalXy/cJabyjlb7JkHli6dYjQ==",
    },
    "v1.0.0-beta.13": {
        "commit": "ba0a56ce72847db6725086ce49d650b8b30fd696",
        "integrity": "sha512-WYSvMgqNE1YkTlX2GudSKLewFfAQJ+0YE31pdXynFoN9e1O/T+tLepG/wrtC0q1+CnEjmol4h2VT0PIwxjy/Pw==",
    },
    "v1.0.0-beta.14": {
        "commit": "6ad7562b6aba175bec616d342e1319e8dbbad1af",
        "integrity": "sha512-TDiY5iVTPnCUQDqQLwhWQo5/o3XMsbRKdKyyDnBECv4iaiEg2qATvboYLF5icKVt489AAr57jfaLkZ66tTVGwg==",
    },
    "v1.0.0-beta.15": {
        "commit": "89a34a44b7a6a5311dc84f3b8a1b8b45c0905aea",
        "integrity": "sha512-I5UB/OWHLGoQW9IVld2yzZFYYiGYNBn2OYRlGbkfj5xDidT26yKYkV7Sr093zJtPh9zbofaLgyT89Ov5U0RRBQ==",
    },
}

HTTPBIN = {
    "v2.14.1": {
        "image": "ghcr.io/mccutchen/go-httpbin:v2.14.1",
        "digest": "sha256:2612f203b1be154524b0dbb65212a158776f0643e2f6e1dad45984fc93413d97",
    },
    "v2.15.0": {
        "image": "ghcr.io/mccutchen/go-httpbin:v2.15.0",
        "digest": "sha256:24528cf5229d0b70065ac27e6c9e4d96f5452a84a3ce4433e56573c18d96827a",
    },
}

DB = {
    "posts": [
        {"id": 1, "title": "alpha", "views": 10},
        {"id": 2, "title": "beta", "views": 20},
        {"id": 3, "title": "gamma", "views": 30},
    ],
    "comments": [{"id": 1, "postId": 1, "body": "first"}],
    "profile": {"name": "frozen-user"},
}


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def run(command: list[str], *, cwd: Path | None = None, timeout: int = 600) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-2000:]}")
    return result.stdout.strip()


def docker_inspect(image: str) -> dict:
    return json.loads(run(["docker", "image", "inspect", image]))[0]


def npm_integrity(path: Path) -> str:
    return "sha512-" + base64.b64encode(sha512(path.read_bytes()).digest()).decode()


def prepare_json_server(version: str, root: Path, base_image: str) -> dict:
    """Download a release tarball, install its locked graph, and build offline."""
    meta = JSON_SERVER[version]
    work = root / version.replace("/", "_")
    work.mkdir(parents=True, exist_ok=True)
    tarball = work / "package.tgz"
    package_version = version.removeprefix("v")
    url = f"https://registry.npmjs.org/json-server/-/json-server-{package_version}.tgz"
    if not tarball.exists():
        with urlopen(Request(url, headers={"User-Agent": "credit-webapi-evolve"}), timeout=60) as response:
            tarball.write_bytes(response.read())
    actual_integrity = npm_integrity(tarball)
    if actual_integrity != meta["integrity"]:
        raise ValueError(f"npm integrity mismatch for {version}: {actual_integrity}")
    package = work / "package.json"
    package.write_text(json.dumps({"private": True, "dependencies": {"json-server": package_version}}, indent=2))
    # npm uses the registry's signed integrity in package-lock.json.  Installation
    # happens on the host; the image build below has networking disabled.
    if not (work / "node_modules" / ".bin" / "json-server").exists():
        run(["npm", "install", "--ignore-scripts", "--omit=dev", "--save-exact"], cwd=work)
    lock = json.loads((work / "package-lock.json").read_text())
    locked = lock["packages"]["node_modules/json-server"]
    if locked["version"] != package_version or locked["integrity"] != meta["integrity"]:
        raise ValueError(f"package lock mismatch for {version}")
    dockerfile = work / "Dockerfile"
    dockerfile.write_text(
        f"FROM {base_image}\nUSER root\nWORKDIR /opt/webapi\n"
        "COPY package.json package-lock.json ./\nCOPY node_modules ./node_modules\n"
        'ENTRYPOINT ["/opt/webapi/node_modules/.bin/json-server"]\n'
    )
    tag = "credit-json-server:" + version.removeprefix("v").replace(".", "-")
    run(["docker", "build", "--network", "none", "-t", tag, "."], cwd=work)
    image = docker_inspect(tag)
    return {
        "version": version, "upstream_tag_commit": meta["commit"],
        "npm_tarball_url": url, "npm_integrity": actual_integrity,
        "npm_tarball_sha256": digest_file(tarball),
        "package_lock_sha256": digest_file(work / "package-lock.json"),
        "dockerfile_sha256": digest_file(dockerfile), "image": tag,
        "image_id": image["Id"], "rootfs_layers": image["RootFS"]["Layers"],
    }


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def container(image: str, command: list[str], *, mount: Path | None = None,
              environment: dict[str, str] | None = None):
    port = available_port()
    args = ["docker", "run", "--rm", "-d", "-p", f"127.0.0.1:{port}:3000"]
    if mount is not None:
        args += ["-v", f"{mount.resolve()}:/data:rw"]
    for key, value in (environment or {}).items():
        args += ["-e", f"{key}={value}"]
    args += [image, *command]
    identifier = run(args)
    base = f"http://127.0.0.1:{port}"
    try:
        last = None
        for _ in range(80):
            try:
                with urlopen(base + "/", timeout=1) as response:
                    response.read(64)
                break
            except Exception as error:  # service startup is expected to race
                last = error
                time.sleep(.1)
        else:
            logs = run(["docker", "logs", identifier])
            raise RuntimeError(f"container failed health check: {last}\n{logs}")
        yield base, identifier
    finally:
        subprocess.run(["docker", "rm", "-f", identifier], capture_output=True)


def request(base: str, path: str, *, method: str = "GET", body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    began = time.perf_counter_ns()
    try:
        response = urlopen(Request(base + path, data=data, headers=headers, method=method), timeout=10)
        raw = response.read()
        status = response.status
        response_headers = dict(response.headers.items())
    except HTTPError as error:
        raw = error.read()
        status = error.code
        response_headers = dict(error.headers.items())
    elapsed = (time.perf_counter_ns() - began) / 1e6
    try:
        parsed = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        parsed = None
    keep = {key.lower(): value for key, value in response_headers.items()
            if key.lower() in {"content-type", "x-total-count", "link", "server-timing", "trailer"}}
    return {"method": method, "path": path, "request_body": body, "status": status,
            "headers": keep, "body_json": parsed,
            "body_text": None if parsed is not None else raw.decode("utf-8", "replace")[:500],
            "body_sha256": digest_bytes(raw), "latency_ms": elapsed}


def classify_json_server(rows: list[dict]) -> dict:
    by_name = {row["probe"]: row for row in rows}
    created = by_name["post_explicit"]["body_json"] or {}
    missing = by_name["missing"]["body_json"]
    limit = by_name["paginate_limit"]["body_json"]
    per_page = by_name["paginate_per_page"]["body_json"]
    automatic = by_name["post_auto"]["body_json"] or {}
    return {
        "client_id_preserved": str(created.get("id")) == "client-choice-42",
        "created_id_type": type(created.get("id")).__name__,
        "automatic_id_type": type(automatic.get("id")).__name__,
        "missing_response_is_json": missing is not None,
        "limit_response_kind": type(limit).__name__,
        "per_page_response_kind": type(per_page).__name__,
        "explicit_id_lookup_status": by_name["get_explicit"]["status"],
    }


def probe_json_server(version: str, artifact: dict, replay: int) -> tuple[list[dict], dict, str]:
    with tempfile.TemporaryDirectory(prefix="credit-webapi-db-") as directory:
        root = Path(directory)
        db = root / "db.json"
        db.write_text(json.dumps(DB, ensure_ascii=False, indent=2))
        restored_hash = digest_file(db)
        if version == "v0.17.4":
            command = ["--watch", "/data/db.json", "--host", "0.0.0.0", "--port", "3000"]
        else:
            command = ["/data/db.json", "--host", "0.0.0.0", "--port", "3000"]
        with container(artifact["image"], command, mount=root) as (base, _):
            specifications = [
                ("list", "GET", "/posts", None),
                ("missing", "GET", "/posts/not-present", None),
                ("paginate_limit", "GET", "/posts?_page=1&_limit=2", None),
                ("paginate_per_page", "GET", "/posts?_page=1&_per_page=2", None),
                ("post_explicit", "POST", "/posts", {"id": "client-choice-42", "title": "delta", "views": 40}),
                ("get_explicit", "GET", "/posts/client-choice-42", None),
                ("post_auto", "POST", "/posts", {"title": "epsilon", "views": 50}),
            ]
            rows = []
            for name, method, path, body in specifications:
                row = request(base, path, method=method, body=body)
                row.update(application="json-server", version=version, replay=replay, probe=name)
                rows.append(row)
        ending_hash = digest_file(db)
    return rows, classify_json_server(rows), restored_hash + ":" + ending_hash


def classify_httpbin(rows: list[dict]) -> dict:
    by_name = {row["probe"]: row for row in rows}
    return {
        "trailers_endpoint_status": by_name["trailers"]["status"],
        "drip_has_server_timing": "server-timing" in by_name["drip"]["headers"],
        "version_status": by_name["version"]["status"],
    }


def probe_httpbin(version: str, meta: dict, replay: int) -> tuple[list[dict], dict]:
    image = docker_inspect(meta["image"])
    repo_digests = image.get("RepoDigests", [])
    if not any(item.endswith("@" + meta["digest"]) for item in repo_digests):
        raise ValueError(f"unexpected OCI digest for {version}: {repo_digests}")
    # go-httpbin listens on 8080 in these images; map it explicitly to the
    # suite's common container port through a tiny docker CLI specialization.
    port = available_port()
    identifier = run(["docker", "run", "--rm", "-d", "-p", f"127.0.0.1:{port}:8080",
                      "-e", "USE_FULL_VERSION=true", meta["image"]])
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(80):
            try:
                request(base, "/version")
                break
            except Exception:
                time.sleep(.1)
        specs = [("version", "/version"), ("trailers", "/trailers"),
                 ("drip", "/drip?duration=0.01s&delay=0s&numbytes=2")]
        rows = []
        for name, path in specs:
            row = request(base, path)
            row.update(application="go-httpbin", version=version, replay=replay, probe=name)
            rows.append(row)
    finally:
        subprocess.run(["docker", "rm", "-f", identifier], capture_output=True)
    return rows, classify_httpbin(rows)


def github_release(repository: str, tag: str) -> dict:
    url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
    with urlopen(Request(url, headers={"Accept": "application/vnd.github+json",
                                      "User-Agent": "credit-webapi-evolve"}), timeout=30) as response:
        payload = json.load(response)
    body = payload.get("body") or ""
    return {"repository": repository, "tag": tag, "url": payload["html_url"],
            "published_at": payload.get("published_at"), "body": body,
            "body_sha256": digest_bytes(body.encode())}


def github_tag(repository: str, tag: str, expected: str) -> dict:
    url = f"https://api.github.com/repos/{repository}/git/ref/tags/{tag}"
    with urlopen(Request(url, headers={"Accept": "application/vnd.github+json",
                                      "User-Agent": "credit-webapi-evolve"}), timeout=30) as response:
        payload = json.load(response)
    actual = payload["object"]["sha"]
    if actual != expected:
        raise ValueError(f"tag {tag} resolved to {actual}, expected {expected}")
    return {"repository": repository, "tag": tag, "commit": actual,
            "url": f"https://github.com/{repository}/tree/{tag}"}


def migration_document() -> dict:
    commit = JSON_SERVER["v1.0.0-beta.15"]["commit"]
    url = f"https://raw.githubusercontent.com/typicode/json-server/{commit}/README.md"
    with urlopen(Request(url, headers={"User-Agent": "credit-webapi-evolve"}), timeout=30) as response:
        text = response.read().decode()
    marker = "## Migration Notes (v0 → v1)"
    start = text.index(marker)
    excerpt = text[start:text.find("\n## ", start + len(marker))].strip()
    return {"repository": "typicode/json-server", "commit": commit,
            "document": "README.md", "url": url, "excerpt": excerpt,
            "document_sha256": digest_bytes(text.encode()),
            "excerpt_sha256": digest_bytes(excerpt.encode())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/webapi_evolve_observed_v1"))
    parser.add_argument("--vendor", type=Path, default=Path("vendor/webapi_evolve/json-server"))
    parser.add_argument("--node-base", default="mereith/langmanus-web:latest")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    args.vendor.mkdir(parents=True, exist_ok=True)

    base = docker_inspect(args.node_base)
    artifacts = {version: prepare_json_server(version, args.vendor, args.node_base)
                 for version in JSON_SERVER}
    evidence = [github_release("typicode/json-server", tag)
                for tag in ("v1.0.0-beta.14", "v1.0.0-beta.15")]
    evidence += [github_release("mccutchen/go-httpbin", tag) for tag in HTTPBIN]
    evidence.append(migration_document())
    tags = [github_tag("typicode/json-server", tag, meta["commit"])
            for tag, meta in JSON_SERVER.items()]

    rows: list[dict] = []
    classifications: dict[str, dict[str, list[dict]]] = {"json-server": {}, "go-httpbin": {}}
    restoration = {}
    for version, artifact in artifacts.items():
        versions = []
        for replay in (1, 2):
            replay_rows, classification, state_hashes = probe_json_server(version, artifact, replay)
            rows += replay_rows
            versions.append(classification)
            restoration[f"json-server:{version}:replay{replay}"] = state_hashes
        classifications["json-server"][version] = versions
    for version, meta in HTTPBIN.items():
        versions = []
        for replay in (1, 2):
            replay_rows, classification = probe_httpbin(version, meta, replay)
            rows += replay_rows
            versions.append(classification)
        classifications["go-httpbin"][version] = versions

    js = classifications["json-server"]
    hb = classifications["go-httpbin"]
    checks = {
        "classification_replay_exact": all(values[0] == values[1]
            for app in classifications.values() for values in app.values()),
        "beta14_ignores_client_ids": (js["v1.0.0-beta.13"][0]["client_id_preserved"]
            and not js["v1.0.0-beta.14"][0]["client_id_preserved"]),
        "beta15_json_404": (not js["v1.0.0-beta.14"][0]["missing_response_is_json"]
            and js["v1.0.0-beta.15"][0]["missing_response_is_json"]),
        "v0_to_v1_pagination_contract": (js["v0.17.4"][0]["limit_response_kind"] == "list"
            and js["v1.0.0-beta.13"][0]["per_page_response_kind"] == "dict"),
        "v0_to_v1_generated_id_type": (js["v0.17.4"][0]["automatic_id_type"] == "int"
            and js["v1.0.0-beta.13"][0]["automatic_id_type"] == "str"),
        "httpbin_trailers_added": (hb["v2.14.1"][0]["trailers_endpoint_status"] == 404
            and hb["v2.15.0"][0]["trailers_endpoint_status"] == 200),
        "httpbin_server_timing_added": (not hb["v2.14.1"][0]["drip_has_server_timing"]
            and hb["v2.15.0"][0]["drip_has_server_timing"]),
    }
    checks["audit_passed"] = all(checks.values())
    protocol = {
        "suite": "WebAPI-Evolve observed upstream release subset",
        "observed_not_injected": True, "live_provider_drift": False,
        "applications": 2, "versions": 6,
        "release_sequence": {"json-server": list(JSON_SERVER), "go-httpbin": list(HTTPBIN)},
        "change_types": ["identifier semantics", "error response schema", "route addition", "response headers"],
        "node_base_image": args.node_base, "node_base_image_id": base["Id"],
        "source_sha256": digest_file(Path(__file__)),
        "database_fixture_sha256": digest_bytes(json.dumps(DB, sort_keys=True).encode()),
        "unknown_or_failed_probe_policy": "fail closed",
        "container_isolation": "one fresh container and restored fixture per release replay",
    }
    (args.output / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.output / "release_artifacts.json").write_text(json.dumps({
        "json-server": artifacts,
        "go-httpbin": {version: {**meta, "image_id": docker_inspect(meta["image"])["Id"]}
                       for version, meta in HTTPBIN.items()},
    }, indent=2))
    (args.output / "release_evidence.json").write_text(json.dumps({"documents": evidence, "tags": tags}, indent=2))
    with (args.output / "responses.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {"checks": checks, "classifications": classifications,
             "restoration_hashes": restoration, "response_rows": len(rows),
             "unique_probes": len(rows) // 2}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    summary = {"audit_passed": checks["audit_passed"], "applications": 2,
               "versions": 6, "unique_probes": len(rows) // 2,
               "documented_changes_reproduced": sum(checks[key] for key in checks if key not in
                   {"classification_replay_exact", "audit_passed"}),
               "classifications": {app: {version: values[0] for version, values in versions.items()}
                                   for app, versions in classifications.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
