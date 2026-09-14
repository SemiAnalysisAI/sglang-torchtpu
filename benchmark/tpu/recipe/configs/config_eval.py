"""Load Python recipes that export a CONFIG dictionary."""

import runpy
from pathlib import Path

ENV_SECTIONS = ("sglang_env", "torch_tpu_env", "runtime_env")
SECTIONS = ("server_args", *ENV_SECTIONS)


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
