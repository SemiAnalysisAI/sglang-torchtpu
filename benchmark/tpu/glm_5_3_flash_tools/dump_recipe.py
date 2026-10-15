"""Print a recipe's resolved server_args and runtime_env as JSON: python dump_recipe.py <module>."""
import importlib, json, sys

cfg = importlib.import_module(sys.argv[1]).CONFIG
print(json.dumps({"server_args": cfg["server_args"], "runtime_env": cfg["runtime_env"]}, indent=2, default=str))
