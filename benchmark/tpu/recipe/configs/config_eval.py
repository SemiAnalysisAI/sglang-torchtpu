"""Load Python recipes that export a CONFIG dictionary."""

import json
import runpy
from pathlib import Path

SECTIONS = ("server_args", "runtime_env")


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Launch config must be an object")
    unknown = config.keys() - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be an object")
    return config


def eval_config(path):
    """Execute a Python recipe and return its validated CONFIG dictionary."""
    path = Path(path).resolve()
    if path.suffix != ".py":
        raise ValueError("Config filename must end in .py")
    namespace = runpy.run_path(str(path))
    if "CONFIG" not in namespace:
        raise ValueError(f"Python recipe must define CONFIG: {path}")
    return validate_config(namespace["CONFIG"])


def server_args_to_cli(settings):
    """Convert final recipe settings to CLI flags; False and None omit a flag."""
    args = []
    for name, value in settings.items():
        flag = "--" + name.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            args.append(flag)
        elif isinstance(value, list):
            args.extend([flag, *map(str, value)])
        elif isinstance(value, dict):
            args.append(f"{flag}={json.dumps(value, allow_nan=False)}")
        else:
            args.append(f"{flag}={value}")
    return args
