"""Convert Python recipe settings to server CLI flags."""

import json


def server_args_to_cli(settings):
    """Convert final settings to CLI flags without modifying the dictionary.

    True emits a switch; False and None omit it. Use strings for options that
    take literal boolean values. Lists emit multiple values; dicts emit JSON.
    """
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
            args.extend([flag, json.dumps(value, allow_nan=False)])
        else:
            args.extend([flag, str(value)])
    return args
