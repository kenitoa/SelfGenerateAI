"""Version and lock-file receipts for the pinned native CPU backend.

The wheel hashes are enforced by pip during deployment. This runtime check
binds later measurements/checkpoints to that lock and the installed versions;
it does not claim to re-hash every installed library binary.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import urllib.parse

from LLM.core import require, file_hash, digest

LOCK_PATH = Path(__file__).resolve().parents[1]/"backend-lock.json"


def _lock(lock_path=None, requirements_path=None, source_path=None):
    path = Path(lock_path) if lock_path is not None else LOCK_PATH
    requirements = Path(requirements_path) if requirements_path is not None else path.with_name("requirements.lock")
    source = Path(source_path) if source_path is not None else path.with_name("requirements.txt")
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2*1024**2,
            "BACKEND_LOCK_MISSING", status=409)
    lock = json.loads(path.read_text("utf-8"))
    require(lock.get("schemaVersion") == 1 and isinstance(lock.get("packages"), dict)
            and 0 < len(lock["packages"]) <= 256, "BACKEND_LOCK_INVALID", status=409)
    target = lock.get("target", {})
    require(target.get("implementation") == "CPython" and target.get("pythonVersion") == "3.12"
            and target.get("system") == "Linux" and target.get("machine") == "x86_64"
            and target.get("device") == "cpu", "BACKEND_LOCK_INVALID", status=409)
    require(requirements.is_file() and not requirements.is_symlink()
            and file_hash(requirements) == lock.get("requirementsSha256"), "BACKEND_REQUIREMENTS_HASH_MISMATCH", status=409)
    require(source.is_file() and not source.is_symlink() and file_hash(source) == lock.get("rootRequirementsSha256"),
            "BACKEND_ROOT_REQUIREMENTS_HASH_MISMATCH", status=409)
    for name, package in lock["packages"].items():
        url = urllib.parse.urlsplit(package.get("wheelUrl", ""))
        host = "download-r2.pytorch.org" if name == "torch" else "files.pythonhosted.org"
        require(package.get("name") == name and isinstance(package.get("version"), str)
                and url.scheme == "https" and url.hostname == host and not url.username and not url.password
                and isinstance(package.get("wheelSha256"), str) and len(package["wheelSha256"]) == 64,
                "BACKEND_LOCK_INVALID", status=409)
        require(not any(word in name for word in ("cuda", "nvidia", "triton")), "CPU_BACKEND_REQUIRED", status=409)
    return path, requirements, lock


def collect(*, lock_path=None, requirements_path=None, source_path=None):
    """Return observed versions plus explicit target/version mismatch lists."""
    path, requirements, lock = _lock(lock_path, requirements_path, source_path)
    versions, issues = {}, []
    for name, package in lock["packages"].items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        versions[name] = version
        if version != package["version"]:
            issues.append(name)
    import torch
    runtime_version = str(torch.__version__)
    cuda_version = torch.version.cuda
    expected_torch = lock["packages"].get("torch", {}).get("version")
    # Some validated Windows distributions use public-version dist-info while
    # their actual loaded CPU library carries +cpu. Record both without hiding it.
    if expected_torch and runtime_version == expected_torch and cuda_version is None:
        if versions.get("torch") in (expected_torch, expected_torch.split("+", 1)[0]):
            issues = [name for name in issues if name != "torch"]
    elif "torch" not in issues:
        issues.append("torch")
    target_issues = []
    implementation = platform.python_implementation()
    python_version = platform.python_version()
    system, machine = platform.system(), platform.machine().lower()
    libc, libc_version = platform.libc_ver()
    if implementation != "CPython": target_issues.append("implementation")
    if sys.version_info[:2] != (3, 12): target_issues.append("pythonVersion")
    if system != "Linux": target_issues.append("system")
    if machine not in ("x86_64", "amd64"): target_issues.append("machine")
    try:
        minimum = tuple(map(int, lock["target"].get("minimumGlibc", "2.28").split(".")))
        if libc != "glibc" or tuple(map(int, libc_version.split("."))) < minimum:
            target_issues.append("minimumGlibc")
    except ValueError:
        target_issues.append("minimumGlibc")
    receipt = {"schemaVersion": 1, "kind": "foundation_backend_receipt",
        "backendLockHash": file_hash(path), "requirementsHash": file_hash(requirements),
        "packageVersions": versions, "torchRuntimeVersion": runtime_version, "cudaVersion": cuda_version,
        "python": {"implementation": implementation, "version": python_version},
        "platform": {"system": system, "machine": machine, "libc": libc, "libcVersion": libc_version},
        "versionMismatches": sorted(issues), "targetMismatches": sorted(set(target_issues)),
        "targetCompatible": not issues and not target_issues,
        "wheelHashesDeclaredInLock": True, "installHashEnforcementObserved": False,
        "installedBinaryHashesReverified": False}
    receipt["receiptHash"] = digest(receipt)
    return receipt


def verify(receipt=None, *, strict_platform=True, lock_path=None, requirements_path=None, source_path=None):
    """Check installed versions; production additionally requires NAS target.

    strict_platform=False is for local diagnostic tests. Its returned receipt
    retains targetCompatible=False and cannot establish production eligibility.
    Passing an earlier receipt also requires exact reproducibility fields.
    """
    current = collect(lock_path=lock_path, requirements_path=requirements_path, source_path=source_path)
    require(not current["versionMismatches"], "BACKEND_VERSION_MISMATCH", status=409,
            packages=current["versionMismatches"])
    require(not strict_platform or not current["targetMismatches"], "BACKEND_TARGET_MISMATCH", status=409,
            fields=current["targetMismatches"])
    if receipt is not None:
        require(isinstance(receipt, dict) and receipt.get("schemaVersion") == 1
                and receipt.get("kind") == "foundation_backend_receipt"
                and receipt.get("receiptHash") == digest({k:v for k,v in receipt.items() if k != "receiptHash"}),
                "BACKEND_RECEIPT_INVALID", status=409)
        fields = ("backendLockHash", "requirementsHash", "packageVersions", "torchRuntimeVersion",
                  "cudaVersion", "python", "platform")
        require(all(receipt.get(key) == current[key] for key in fields), "BACKEND_RECEIPT_MISMATCH", status=409)
    return current
