"""Check source/ZIP contents against the reviewed public manifest.

Uses only the Python standard library. Findings contain paths/rule names,
never matched credential values.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "PUBLIC_MANIFEST.json"
IGNORED_LOCAL = {".git", "__pycache__", ".pytest_cache", ".venv", ".venv-copilot"}
PRIVATE_DIRS = {"model", "models", "data", "replays", "牌谱爬取", "browser_data",
                "mitm_config", "log", "logs", "training_runs", "ppo_runs", "artifacts",
                "release", "shangrao_runs", "feature_runs", ".aws", ".ssh", ".codex"}
PRIVATE_SUFFIXES = {".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".safetensors", ".onnx",
                    ".npy", ".npz", ".f32", ".u8", ".i16", ".log", ".jsonl", ".ndjson",
                    ".pem", ".key", ".crt", ".cer", ".p12", ".pfx", ".pcap", ".pcapng",
                    ".har", ".mitm", ".db", ".sqlite", ".zip", ".gz", ".7z", ".psd",
                    ".pyd", ".dll", ".so", ".exe"}
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b"),
    "personal-windows-path": re.compile(r"[A-Za-z]:[/\\](?:Users|code)[/\\]", re.I),
    "credential-in-url": re.compile(r"https?://[^\s/@:'\"]+:[^\s/@'\"]+@"),
}
SENSITIVE_NAME = re.compile(r"(?:^|_)(?:password|passwd|secret|apikey|api_key|access_token|auth_token)$", re.I)
# These exact upstream localization strings label input boxes; they are not credentials.
UI_CREDENTIAL_LABELS = {
    "AKAGI_OT_APIKEY": {"AkagiOT API Key"},
    "MJAPI_SECRET": {"MJAPI Secret", "MJAPI 密钥"},
}


def check_file(name, payload, findings):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        findings.append((name, "invalid archive path"))
    if any(part in PRIVATE_DIRS for part in path.parts):
        findings.append((name, "private directory"))
    if path.suffix.lower() in PRIVATE_SUFFIXES:
        findings.append((name, "private file type"))
    if path.name in {"settings.json", "progress.json", "output.txt", "autodl_remote.py"} or path.name.startswith((".env", "fetch_", "id_rsa", "id_ed25519")):
        findings.append((name, "private runtime/configuration file"))
    if path.suffix.lower() not in {".py", ".json", ".md", ".csv", ".txt", ".proto"}:
        return
    try:
        source = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        findings.append((name, "invalid UTF-8 text"))
        return
    for rule, pattern in PATTERNS.items():
        if pattern.search(source):
            findings.append((name, rule))
    if path.suffix == ".py":
        try:
            tree = ast.parse(source, filename=name)
        except SyntaxError:
            findings.append((name, "Python syntax error"))
            return
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            value = getattr(node, "value", None)
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
                for target in targets:
                    label = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ""
                    if SENSITIVE_NAME.search(label):
                        if name == "integrations/MahjongCopilot/common/lan_str.py" and value.value in UI_CREDENTIAL_LABELS.get(label, set()):
                            continue
                        findings.append((name, "nonempty credential literal"))
    if path.suffix == ".json":
        try:
            data = json.loads(source)
        except json.JSONDecodeError:
            findings.append((name, "invalid JSON"))
            return
        def inspect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if SENSITIVE_NAME.search(key) and isinstance(item, str) and item:
                        findings.append((name, "nonempty JSON credential"))
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)
        inspect(data)
        if path.parts[0] == "battle_results":
            if path.name == "battle_report.json":
                if not data.get("rankings") or any(row.get("seed_groups", 0) < 1000 for row in data["rankings"]):
                    findings.append((name, "battle seed groups below 1000"))
                if any(row.get("paired_seed_groups", 0) < 1000 for row in data.get("pairwise", [])):
                    findings.append((name, "paired battle seed groups below 1000"))
            elif path.name == "index.json":
                if any(row.get("seed_groups", 0) < 1000 for row in data):
                    findings.append((name, "battle index seed groups below 1000"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Verify a ZIP instead of the source directory")
    args = parser.parse_args()
    findings = []
    if args.archive:
        files = {}
        with zipfile.ZipFile(args.archive) as archive:
            entries = [i for i in archive.infolist() if not i.is_dir()]
            prefixes = {PurePosixPath(i.filename).parts[0] for i in entries}
            prefix = next(iter(prefixes)) + "/" if len(prefixes) == 1 and all("/" in i.filename for i in entries) else ""
            for entry in entries:
                name = entry.filename.removeprefix(prefix)
                if name in files:
                    findings.append((name, "duplicate ZIP entry"))
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    findings.append((name, "symlink"))
                files[name] = archive.read(entry)
    else:
        files = {}
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if any(p in IGNORED_LOCAL or p.startswith(".venv-") for p in relative.parts):
                continue
            if path.is_symlink():
                findings.append((relative.as_posix(), "symlink"))
            elif path.is_file():
                files[relative.as_posix()] = path.read_bytes()
    if MANIFEST not in files:
        print("FAIL: public manifest is missing")
        return 1
    manifest = json.loads(files[MANIFEST].decode("utf-8"))
    expected = {entry["path"]: entry for entry in manifest["files"]}
    for name in sorted(set(files) - set(expected) - {MANIFEST}):
        findings.append((name, "file not in reviewed manifest"))
    for name in sorted(set(expected) - set(files)):
        findings.append((name, "reviewed file missing"))
    for name, payload in files.items():
        check_file(name, payload, findings)
        if name in expected:
            entry = expected[name]
            if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                findings.append((name, "content changed since review"))
    for name, rule in sorted(set(findings)):
        print(f"FAIL: {name}: {rule}")
    if findings:
        return 1
    print(f"PASS: {len(files)} public files; manifest, integrity and sensitive-file checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
