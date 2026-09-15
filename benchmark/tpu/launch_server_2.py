"""Launch SGLang with ordinary CLI flags and an optional Python recipe.

python -m benchmark.tpu.launch_server_2 --config benchmark/tpu/recipe/configs/qwen3_0_6b.py

Explicit CLI flags follow recipe defaults. Runtime environment variables are
applied before importing SGLang in the server process. Tests can override a
recipe dictionary, convert its server_args with server_args_to_cli, and pass
those flags through other_args and runtime_env through env to popen_launch_server.
"""

import argparse
import json
import os
import re
import sys

from benchmark.tpu.dump_meta import _looks_secret, _redact_secrets
from benchmark.tpu.recipe.configs.config_eval import (
    eval_config,
    server_args_to_cli,
    validate_config,
)


def build_launch(config, extra_args=(), inherited=None):
    validate_config(config)
    env = dict(os.environ if inherited is None else inherited)
    for name, value in config.get("runtime_env", {}).items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            raise ValueError(f"Invalid environment variable: {name}")
        if value is None:
            env.pop(name, None)
        elif isinstance(value, str) and "\0" not in value:
            env[name] = value
        else:
            raise ValueError(f"{name} must be a string without NUL, or null")
    return [
        sys.executable,
        "-u",
        "-m",
        "sglang.launch_server",
        *server_args_to_cli(config.get("server_args", {})),
        *extra_args,
    ], env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", help="Python recipe defining CONFIG")
    parser.add_argument("--dry-run", action="store_true")
    options, extra_args = parser.parse_known_args(argv)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    try:
        config = eval_config(options.config) if options.config else {}
        command, env = build_launch(config, extra_args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if options.dry_run:
        redacted = []
        secret = False
        for arg in command:
            if arg.startswith("--"):
                name, separator, _ = arg.partition("=")
                secret = _looks_secret(name)
                redacted.append(name + "=<redacted>" if secret and separator else arg)
            else:
                redacted.append("<redacted>" if secret else arg)
        print(
            json.dumps(
                {
                    "argv": redacted,
                    "runtime_env": _redact_secrets(config.get("runtime_env", {})),
                },
                indent=2,
            )
        )
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
