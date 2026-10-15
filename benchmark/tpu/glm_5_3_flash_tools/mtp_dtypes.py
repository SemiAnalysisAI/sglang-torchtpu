"""Print dtype/shape of every checkpoint tensor in layer 45 (the MTP head), skipping routed experts past 0."""
import json, os, re, sys
from safetensors import safe_open

root = sys.argv[1] if len(sys.argv) > 1 else "/weights/GLM-5.3-Flash"
idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
names = sorted(n for n in idx if n.startswith("model.layers.45.") and not re.search(r"experts\.([1-9]\d*)\.", n))
by_file = {}
for n in names:
    by_file.setdefault(idx[n], []).append(n)
for f, ns in by_file.items():
    with safe_open(os.path.join(root, f), "pt") as h:
        for n in ns:
            s = h.get_slice(n)
            print(n, s.get_dtype(), s.get_shape())
cfg = json.load(open(os.path.join(root, "config.json")))
print("quant ignore:", [m for m in (cfg.get("quantization_config", {}).get("modules_to_not_convert") or cfg.get("quantization_config", {}).get("ignored_layers") or []) if "45" in m or "attn" in m][:40])
