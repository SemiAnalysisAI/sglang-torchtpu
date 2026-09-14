"""Opt-in TPU launcher: config -> CLI arguments and child environment -> exec.

python benchmark/tpu/launch_server_2.py --config benchmark/tpu/recipe/configs/qwen3_0_6b.py
python benchmark/tpu/launch_server_2.py --config recipe.py -- --port 30001

Python recipes export CONFIG and can import shared defaults directly.

server_args keys are canonical CLI names without '--'. True emits a switch;
false/null omit it; arrays emit multiple arguments; objects emit JSON values.
Use strings for options that take literal boolean values. Explicit flags after
'--' take precedence; use canonical names when overriding config entries.

Environment sections override inherited variables; null removes a variable.
All runtime settings are explicit: this launcher does not infer topology, set
up caches, build telemetry libraries, or source the legacy shell environment.
"""

import argparse
import json
import math
import os
import re
import sys

from dump_meta import _looks_secret, _redact_secrets
from recipe.configs.config_eval import (
    ENV_SECTIONS,
    eval_config,
    validate_config,
)


def cli_args(settings):
    args = []
    for name, value in settings.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name) or name == "config":
            raise ValueError(f"Expected a canonical CLI name without '--': {name}")
        if value is None or value is False:
            continue
        flag = f"--{name}"
        if value is True:
            args.append(flag)
        elif isinstance(value, list):
            if not value or any(type(item) not in (str, int, float) for item in value):
                raise ValueError(f"{name} requires a nonempty array of scalar values")
            json.dumps(value, allow_nan=False)
            args.extend([flag, *map(str, value)])
        elif isinstance(value, dict):
            args.append(
                f"{flag}={json.dumps(value, separators=(',', ':'), allow_nan=False)}"
            )
        elif isinstance(value, (str, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite value for {name}")
            if isinstance(value, str) and "\0" in value:
                raise ValueError(f"NUL in CLI value for {name}")
            args.append(f"{flag}={value}")
        else:
            raise ValueError(f"Unsupported CLI value for {name}")
    if any("\0" in arg for arg in args):
        raise ValueError("NUL in CLI arguments")
    return args


def build_launch(config, overrides=(), inherited=None):
    validate_config(config)
    env = dict(os.environ if inherited is None else inherited)
    for section in ENV_SECTIONS:
        for name, value in config.get(section, {}).items():
            expected = (
                "sglang_env"
                if name.startswith("SGLANG_")
                else "torch_tpu_env" if name.startswith("TORCH_TPU_") else "runtime_env"
            )
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) or expected != section:
                raise ValueError(
                    f"Invalid variable in {section}: {name} (expected {expected})"
                )
            if value is None:
                env.pop(name, None)
            elif isinstance(value, str) and "\0" not in value:
                env[name] = value
            else:
                raise ValueError(f"{name} must be a string without NUL, or null")
    settings = config.get("server_args", {})
    # Validate even overridden entries so typos do not silently survive in recipes.
    cli_args(settings)
    names = {arg[2:].split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    args = cli_args(
        {name: value for name, value in settings.items() if name not in names}
    )
    return [sys.executable, "-u", "-m", "sglang.launch_server", *args, *overrides], env


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv[:split])
    try:
        config = eval_config(options.config)
        command, env = build_launch(config, argv[split + 1 :])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if options.dry_run:
        public_command = list(command)
        for index, arg in enumerate(command):
            key, separator, _ = arg.partition("=")
            if arg.startswith("--") and _looks_secret(key):
                if separator:
                    public_command[index] = key + "=***redacted***"
                elif index + 1 < len(command):
                    public_command[index + 1] = "***redacted***"
        print(
            json.dumps(
                {
                    "argv": public_command,
                    **_redact_secrets(
                        {section: config.get(section, {}) for section in ENV_SECTIONS}
                    ),
                },
                indent=2,
            )
        )
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
