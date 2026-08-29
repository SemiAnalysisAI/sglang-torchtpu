#!/usr/bin/env python3
"""Discover registered TPU tests through the unified TPU_LAUNCH schema."""

import argparse
import ast
import glob
import json
import os
import re
import shlex
import sys


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _registration(node):
    if not (
        isinstance(node, ast.Call) and getattr(node.func, "id", "") == "register_tpu_ci"
    ):
        return None
    keywords = {item.arg: item.value for item in node.keywords}
    return {
        "suite": _literal(keywords.get("suite")),
        "nightly": _literal(keywords.get("nightly")) or False,
        "disabled": _literal(keywords.get("disabled")),
    }


def _scan(root, suite, mode, nightly_only):
    found = {}
    for path in glob.glob(f"{root}/**/*.py", recursive=True):
        try:
            with open(path) as source:
                tree = ast.parse(source.read())
        except (OSError, SyntaxError):
            continue
        registrations = []
        launch = None
        for node in ast.walk(tree):
            registration = _registration(node)
            if registration:
                registrations.append(registration)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TPU_LAUNCH"
                for target in node.targets
            ):
                launch = _literal(node.value)
        matching = [
            registration
            for registration in registrations
            if registration["suite"] == suite
            and (registration["nightly"] or not nightly_only)
        ]
        if not matching:
            continue
        if any(registration["disabled"] is not None for registration in matching):
            reason = next(
                registration["disabled"]
                for registration in matching
                if registration["disabled"] is not None
            )
            print(f"skip {path}: disabled ({reason})", file=sys.stderr)
            continue
        if not isinstance(launch, dict) or launch.get("mode") != mode:
            raise ValueError(f"{path}: {suite} test needs TPU_LAUNCH mode={mode!r}")
        found[path] = launch
    return found


def _multihost_entries(found):
    groups = {}
    for path, launch in sorted(found.items()):
        try:
            script = launch["launch_script"]
            hosts = int(launch.get("hosts", 2))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid TPU_LAUNCH {launch!r}") from exc
        weights = launch.get("weights_pvc", "") or ""
        key = (script, weights, hosts)
        entry = groups.setdefault(key, {"files": [], "timeout_per_file": 3600})
        entry["files"].append(path)
        entry["timeout_per_file"] = max(
            entry["timeout_per_file"], int(launch.get("timeout_per_file", 3600))
        )
    output = []
    for (script, weights, hosts), entry in groups.items():
        base = os.path.basename(script).lower().removesuffix(".sh").removesuffix("_mh")
        slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
        output.append(
            {
                "name": f"mh-{slug}",
                "launch_script": script,
                "weights_pvc": weights,
                "hosts": hosts,
                "files": " ".join(sorted(entry["files"])),
                "timeout_per_file": entry["timeout_per_file"],
            }
        )
    return sorted(output, key=lambda entry: entry["name"])


def _pd_entry(path, launch):
    try:
        name = launch["name"]
        prefill_hosts = int(launch["prefill_hosts"])
        decode_hosts = int(launch["decode_hosts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid TPU_LAUNCH {launch!r}") from exc
    if not isinstance(name, str) or not name or prefill_hosts < 1 or decode_hosts < 1:
        raise ValueError(f"{path}: invalid TPU_LAUNCH {launch!r}")
    env = launch.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise ValueError(f"{path}: TPU_LAUNCH env must map strings to strings")
    command = ["env", *(f"{key}={value}" for key, value in sorted(env.items()))]
    if not env:
        command = []
    command.extend(["python3", path])
    args = launch.get("args", "")
    if not isinstance(args, str):
        raise ValueError(f"{path}: TPU_LAUNCH args must be a string")
    command.extend(shlex.split(args))
    return {
        "name": name,
        "prefill_hosts": prefill_hosts,
        "decode_hosts": decode_hosts,
        "hosts": max(prefill_hosts, decode_hosts),
        "exec_cmd": shlex.join(command),
        "files": path,
        "weights_pvc": launch.get("weights_pvc", "") or "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--mode", choices=("multihost", "pd"), required=True)
    parser.add_argument("--root", default="test/registered")
    parser.add_argument("--only", default="all")
    parser.add_argument(
        "--nightly", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    found = _scan(args.root, args.suite, args.mode, args.nightly)
    entries = (
        _multihost_entries(found)
        if args.mode == "multihost"
        else sorted(
            (_pd_entry(path, launch) for path, launch in found.items()),
            key=lambda entry: entry["name"],
        )
    )
    if args.only != "all":
        entries = [entry for entry in entries if entry["name"] == args.only]
    print(json.dumps(entries, separators=(",", ":")))


if __name__ == "__main__":
    main()
