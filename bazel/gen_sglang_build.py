#!/usr/bin/env python3
"""Generate fine-grained BUILD files for python/sglang.

Layout mirrors torch_tpu's own Bazel build: one py_library per package
directory plus one py_library per submodule, with deps resolved from an AST
import scan against three sources -- sibling first-party targets, the pip
hub's extracted wheels (site-packages dirs), and the stdlib.

sglang's import graph is genuinely cyclic (mutual imports between modules,
mostly through registry/__init__ re-export chains), which Bazel cannot
represent, so the generator collapses each strongly connected component into
one py_library and points every dep into the component at that target;
acyclic modules keep their own fine-grained targets. A component's py_library
gathers its sources through per-directory filegroups (one ``cycle_N_files``
per directory holding members): a glob may not cross a package boundary, and
every member directory is its own package. The filegroups feed ``srcs``, not
``deps``, so they add no dependency edges and the collapsed target stays
acyclic. Component sizes are printed on regen so growth of the big one is
visible.

The output is committed to the repo, exactly like
requirements/requirements_lock_3_12.txt: the build consumes it; regenerating
is a dev action, never part of a bazel invocation. Run after adding/removing
modules or changing their imports:

    bazel run //bazel:gen_sglang_build
    # or directly, once any bazel build has materialized the pip hub:
    .venv/bin/python bazel/gen_sglang_build.py

It rewrites the per-directory BUILD files and ``python/cycles.bzl`` (the
collapsed cycle targets plus the ``ALL_MODULE_DEPS`` list the handwritten
``python/BUILD.bazel`` builds its ``:sglang`` aggregator from).

Only the TPU closure gets BUILD files: directories reachable from the launch
entry points by static imports, plus DYNAMIC_ROOTS below. The rest of the
tree (CUDA kernel zoos, multimodal generation, ...) stays Bazel-free, which
keeps the change non-invasive.
"""

# Directories whose modules are loaded dynamically (importlib/pkgutil) and
# therefore invisible to the static import scan, but required on the TPU
# serving path:
# - srt/models: the model registry pkgutil-walks and imports EVERY module in
#   the directory at startup (import failures are logged and skipped, so a
#   missing BUILD file would not crash -- it would silently drop the model).
#   Model files import their multimodal processors and attention backends,
#   which pulls in the rest of srt/multimodal and srt/layers/attention.
# - srt/speculative: spec-decoding algorithms are selected by name at launch.
# - kernels/ops: model files and multimodal processors import kernel ops by
#   name across the subtree (attention backends, mm processors).
# - hardware_backend: layers/quantization unconditionally imports sibling
#   backends' quantization helpers (upstream coupling, not TPU-specific), so
#   the whole tree rides along.
# - layers/quantization: quantization methods register their schemes by name
#   at import time via dynamic loads.
# Everything under these roots gets BUILD files; nothing else outside the
# static closure does.
DYNAMIC_ROOTS = (
    "sglang/srt/models",
    "sglang/srt/speculative",
    "sglang/srt/multimodal",
    "sglang/srt/layers/attention",
    "sglang/srt/layers/quantization",
    "sglang/srt/hardware_backend",
    "sglang/kernels/ops",
)

# Static roots of the closure: the launch entry point, the bench entry, the
# package root, and the test harness. The e2e test file imports
# sglang.test.ci directly and the TPU harness uses sglang.test.kits, so the
# whole sglang.test tree -- kits, fixtures, runners -- is in the closure.
STATIC_ROOTS = ("sglang.launch_server", "sglang.bench_serving", "sglang")

# Directory prefixes treated as static roots too (the static scan starts from
# module names, which cannot express a package prefix).
STATIC_DIR_ROOTS = ("sglang/test",)

# Modules outside the closure that closure modules import anyway (diffusion
# kernels pull multimodal_gen; the TPU harness pulls two test fixtures and the
# tiktoken tokenizer). The TPU serving path never executes these imports --
# they sit in GPU-only or eval-only code -- so the modules are simply absent
# from the Bazel closure, and the import would fail loudly if ever reached.
# The generator errors on any OTHER non-closure dep so a real leak is caught
# at regen time, not at runtime. Values are module names; "pkg.*" allows the
# whole subtree (multimodal_gen: diffusion serving, never on the TPU path).
NON_CLOSURE_DEPS = {
    "sglang.multimodal_gen.*",
    "sglang.srt.tokenizer.tiktoken_tokenizer",
    "sglang.test.server_fixtures.disaggregation_fixture",
    "sglang.test.server_fixtures.streaming_session_fixture",
}


def _is_non_closure_allowed(mod: str) -> bool:
    for pat in NON_CLOSURE_DEPS:
        if pat.endswith(".*"):
            if mod == pat[:-2] or mod.startswith(pat[:-1]):
                return True
        elif mod == pat:
            return True
    return False


def closure_dirs(g: "Graph"):
    """Directory set that gets BUILD files: static import reachability from
    STATIC_ROOTS, plus DYNAMIC_ROOTS, plus the package ancestors of every
    reachable module (importing sglang.a.b executes sglang/__init__ and
    sglang/a/__init__, and those directories own pkg targets / resources)."""
    reachable = set()
    work = [m for m in STATIC_ROOTS if m in g.modules]
    while work:
        mod = work.pop()
        if mod in reachable:
            continue
        reachable.add(mod)
        work.extend(d for d in g.edges.get(mod, ()) if d not in reachable)
    for root in DYNAMIC_ROOTS + STATIC_DIR_ROOTS:
        root_dir = g.src_root / root
        if root_dir.is_dir():
            reachable.update(
                m for m in g.modules
                if g.pkg_dir(m) == Path(root) or Path(root) in g.pkg_dir(m).parents
            )
    dirs = set()
    for m in reachable:
        pkg = g.pkg_dir(m)
        while pkg.parts:
            dirs.add(pkg)
            pkg = pkg.parent
    return dirs

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

GENERATED = "# GENERATED FILE -- do not edit. Regenerate: bazel run //bazel:gen_sglang_build"
PY_LOAD = 'load("@rules_python//python:defs.bzl", "py_library")'
# Modules the import scan tolerates without a BUILD dep: optional integrations
# loaded conditionally (try/except or env-gated hardware paths) and never
# present in the TPU lock. Reported, not an error.
KNOWN_UNRESOLVED = {
    "aiter", "compressed_tensors_offline", "custom_ops", "decord", "deep_gemm",
    "flash_attn", "flash_mla", "flashinfer", "flexkv", "lmcache", "mindspore",
    "mlx", "moe_wna16", "mooncake", "mori", "nixl", "pyhip", "pynvml",
    "quant_util", "sgl_kernel", "torch_musa", "torch_npu", "torch_xpu",
    "triton_kernels", "umbp", "vllm", "xformers",
}

# Imports resolved outside the sglang pip hub. jax/libtpu must come from
# torch_tpu's own hub so the versions stay locked to the libtpu/torch the
# source build was compiled against -- the sglang lock deliberately excludes
# the TPU stack (see requirements/tpu_stack_excludes.txt).
EXTRA_HUB_LABELS = {
    "jax": "@torch_tpu//shims/jax:pallas",
    "torch_tpu": "@torch_tpu//torch_tpu",
}

# Non-Python resource trees some packages resolve from the filesystem at
# import time (importlib.find_spec(...).parent / "include", .../ "csrc"), e.g.
# sglang.kernels.jit's KERNEL_PATH. The generated py_library srcs carry only
# *.py, so these must ride along as runfiles data or the import dies with
# "Cannot find sglang.kernels.jit path" under Bazel.
RESOURCE_SUBDIRS = ("include", "csrc")

STDLIB = set(sys.stdlib_module_names)


def module_name(src_root: Path, path: Path) -> str:
    rel = path.relative_to(src_root).with_suffix("")
    return ".".join(p for p in rel.parts if p != "__init__")


def file_imports(path: Path):
    tree = ast.parse(path.read_bytes())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # `from pkg import sub` binds pkg.sub when it is a submodule, so the
            # import touches both; recording only pkg would miss the edge.
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def hub_top_level_modules(external: Path, hub: str):
    """import-name -> normalized package name, from the hub's extracted wheels."""
    mapping = {}
    prefix = f"rules_python++pip+{hub}_312_"
    repos = sorted(external.glob(f"{prefix}*"))
    if not repos:
        raise SystemExit(
            f"no {prefix}* repos under {external}; run any bazel build "
            "first so the hub materializes"
        )
    for repo in repos:
        pkg = repo.name[len(prefix):]
        sp = repo / "site-packages"
        if not sp.is_dir():
            continue
        for entry in sp.iterdir():
            if entry.suffix in (".dist-info", ".data"):
                continue
            name = entry.stem if entry.suffix == ".py" else entry.name
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                mapping.setdefault(name, pkg)
    return mapping


def tarjan_scc(nodes, edges):
    index, low, on_stack, stack, sccs = {}, {}, set(), [], []
    counter = [0]

    def visit(v0):
        index[v0] = low[v0] = counter[0]
        counter[0] += 1
        stack.append(v0)
        on_stack.add(v0)
        work = [(v0, iter(edges.get(v0, ())))]
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(edges.get(w, ()))))
                    advanced = True
                    break
                elif w in on_stack:
                    low[v] = min(low[v], index[w])
            if not advanced:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.append(w)
                        if w == v:
                            break
                    if len(scc) > 1:
                        sccs.append(scc)

    for v in nodes:
        if v not in index:
            visit(v)
    return sccs


class Graph:
    """The resolved module graph plus target naming for one sglang tree."""

    def __init__(self, src_root: Path, hub: str, hub_modules):
        self.src_root = src_root
        self.hub = hub
        self.modules = {}
        for f in sorted((src_root / "sglang").rglob("*.py")):
            self.modules[module_name(src_root, f)] = f

        self.edges = defaultdict(set)      # module -> first-party module deps
        self.ext_deps = defaultdict(set)   # module -> pip package names
        self.unknown = defaultdict(set)
        for mod, path in sorted(self.modules.items()):
            for name in file_imports(path):
                top = name.split(".")[0]
                if top == "sglang":
                    # The module and every package prefix on the path to it
                    # are all executed by the import.
                    cur = name
                    while cur:
                        dep = self.resolve(cur)
                        if dep and dep != mod:
                            self.edges[mod].add(dep)
                        cur = cur.rpartition(".")[0]
                elif top in STDLIB:
                    continue
                elif top in hub_modules:
                    self.ext_deps[mod].add(self.pip_label(hub_modules[top]))
                elif top in EXTRA_HUB_LABELS:
                    self.ext_deps[mod].add(EXTRA_HUB_LABELS[top])
                elif top not in KNOWN_UNRESOLVED:
                    self.unknown[top].add(mod)

        self.sccs = tarjan_scc(list(self.modules), self.edges)
        self.sccs.sort(key=lambda s: (-len(s), s[0]))
        self.scc_of = {m: i for i, s in enumerate(self.sccs) for m in s}
        self.keep_dirs = None  # set by main() before rendering

        # Packages holding a direct resource tree (RESOURCE_SUBDIRS), plus the
        # module-name prefix a module must have to live in that package's
        # subtree. A resource at sglang/kernels/jit is found by any import
        # under sglang.kernels.jit.*, so a cycle reaching that subtree needs
        # the tree in its runfiles.
        self.pkgs_with_resources = defaultdict(list)
        self.pkg_module_prefix = {}
        for pkg_dir in sorted({self.pkg_dir(m) for m in self.modules}):
            for sub in RESOURCE_SUBDIRS:
                if (self.src_root / pkg_dir / sub).is_dir():
                    self.pkgs_with_resources[pkg_dir].append(sub)
            self.pkg_module_prefix[pkg_dir] = ".".join(pkg_dir.parts) + "."

    def subtree_touched_by_cycle(self, pkg_dir: Path) -> bool:
        """True when any cycle member lives in this package's subtree -- i.e.
        its resource trees belong on that cycle target, not this package's."""
        prefix = self.pkg_module_prefix[pkg_dir]
        return any(m.startswith(prefix) for m in self.scc_of)

    def droppable(self, mod: str) -> bool:
        """True when `mod` is outside the emitted closure and may be dropped
        from deps (allowlisted; never reached on the TPU serving path)."""
        return (self.keep_dirs is not None
                and self.pkg_dir(mod) not in self.keep_dirs
                and _is_non_closure_allowed(mod))

    def resolve(self, name: str):
        m = name
        while m and m not in self.modules:
            m = m.rpartition(".")[0]
        return m or None

    def anchor(self, mod: str) -> str:
        """The module whose target owns `mod` (itself, or its component's)."""
        if mod in self.scc_of:
            members = self.sccs[self.scc_of[mod]]
            return min(members, key=lambda m: (m.count("."), m))
        return mod

    def pkg_dir(self, mod: str) -> Path:
        return self.modules[mod].parent.relative_to(self.src_root)

    def label_of(self, mod: str) -> str:
        """The py_library providing `mod` (only called for non-cycle modules)."""
        path = self.modules[mod]
        pkg = path.parent.relative_to(self.src_root)
        if mod in self.scc_of:
            return self.files_label_of(mod)
        name = "pkg" if path.stem == "__init__" else path.stem
        return f"//{self.src_root}/{pkg}:{name}"

    def has_own_target(self, mod: str) -> bool:
        """True when `mod` is emitted as its own per-dir target (not a cycle
        member folded into a cycle_N target)."""
        return mod not in self.scc_of

    def files_label_of(self, mod: str) -> str:
        """The label a consumer uses for a cycle member: the cycle's target.

        Every cycle collapses to one ``sglang_cycle`` target in ``//python``,
        so all members share that single label. Verified acyclic: no external
        dep of any cycle reaches back into the cycle.
        """
        return f"//{self.src_root}:cycle_{self.scc_of[mod]}"

    def dep_label(self, mod: str) -> str:
        """What a *consumer* of `mod` should put in deps."""
        if mod in self.scc_of:
            return self.files_label_of(mod)
        return self.label_of(mod)

    def pip_label(self, pkg: str) -> str:
        return f"@{self.hub}//{pkg}"

    def imports_attr(self, pkg: Path) -> str:
        # sglang/... files import as `sglang....`: the imports root is the dir
        # holding the package (python/), relative to the BUILD file's dir.
        depth = len(pkg.parts)  # e.g. sglang/srt -> 2
        return "/".join([".."] * depth)


def render_dir(g: Graph, pkg_dir: Path, mods_in_dir) -> str:
    lines = [GENERATED, PY_LOAD, "", 'package(default_visibility = ["//visibility:public"])', ""]
    imports = g.imports_attr(pkg_dir)

    def first_party_deps(mod: str):
        """Dep labels for one module, deduped: when both a package and one of
        its members are imported, only the member's target is needed (the
        member's own deps carry the rest of the package) -- unless the package
        has no pkg target of its own (its __init__ is itself a cycle member),
        in which case the package dep is a distinct cycle edge and stays."""
        raw = {d for d in g.edges.get(mod, ()) if not g.droppable(d)}
        keep = set(raw)
        for d in raw:
            parent = d.rpartition(".")[0]
            if parent in raw and parent != d and g.has_own_target(parent):
                keep.discard(parent)
        return {g.dep_label(d) for d in sorted(keep)}

    def dep_block(deps):
        out = ["    deps = ["]
        out += [f'        "{d}",' for d in sorted(deps)]
        out.append("    ],")
        return out

    singletons = []
    pkg_init = None
    scc_members = defaultdict(list)
    for mod in sorted(mods_in_dir):
        if mod in g.scc_of:
            scc_members[g.scc_of[mod]].append(mod)
            continue
        if g.modules[mod].stem == "__init__":
            pkg_init = mod
        else:
            singletons.append(mod)

    # Cycle members become one glob'd filegroup per cycle in this directory,
    # gathered into the ``cycle_N`` py_library that //python emits for the
    # whole cycle (see render_cycles). A glob cannot cross a package boundary,
    # so the collection has to start here. The filegroups feed ``srcs``, not
    # ``deps``: they add no dependency edges, so the collapsed cycle target
    # stays acyclic (verified: no external dep of any cycle reaches back).
    for scc_idx, members in sorted(scc_members.items()):
        files = sorted(g.modules[m].name for m in members)
        lines += ["filegroup(", f'    name = "cycle_{scc_idx}_files",',
                  "    srcs = glob(["]
        lines += [f'        "{f}",' for f in files]
        lines += ["    ]),", ")", ""]

    # Non-Python resource trees (RESOURCE_SUBDIRS) this package resolves from
    # the filesystem at import time ride along as runfiles data. Emitted for
    # trees directly under this package's dir only (a parent's glob cannot
    # cross a subpackage's BUILD file). The consumer depends on the cycle: a
    # package whose subtree a cycle reaches hands its trees to that cycle
    # (render_cycles); an untouched package attaches them to its own pkg target
    # below.
    abs_dir = g.src_root / pkg_dir
    my_resources = [sub for sub in RESOURCE_SUBDIRS if (abs_dir / sub).is_dir()]
    for sub in my_resources:
        lines += ["filegroup(", f'    name = "{sub}_data",',
                  f'    srcs = glob(["{sub}/**"]),', ")", ""]

    if pkg_init is not None:
        deps = first_party_deps(pkg_init)
        deps.update(g.ext_deps.get(pkg_init, ()))
        lines += ["py_library(", '    name = "pkg",', '    srcs = ["__init__.py"],',
                  f'    imports = ["{imports}"],']
        if deps:
            lines += dep_block(deps)
        # Attach this package's resource trees when no cycle reaches into its
        # subtree (otherwise render_cycles already put them on the cycle).
        if my_resources and not g.subtree_touched_by_cycle(pkg_dir):
            lines += ["    data = ["] + [f'        ":{sub}_data",' for sub in my_resources] + ["    ],"]
        lines += [")", ""]

    for mod in singletons:
        path = g.modules[mod]
        deps = first_party_deps(mod)
        deps.update(g.ext_deps.get(mod, ()))
        lines += ["py_library(", f'    name = "{path.stem}",', f'    srcs = ["{path.name}"],',
                  f'    imports = ["{imports}"],']
        if deps:
            lines += dep_block(deps)
        lines += [")", ""]

    return "\n".join(lines)


def render_cycles(g: Graph, keep_dirs=None) -> str:
    """//python:cycles.bzl -- one ``declare_cycles`` macro emitting one glob'd
    py_library per import cycle.

    Kept in a ``.bzl`` rather than inline in ``python/BUILD.bazel`` so the
    BUILD file can stay handwritten: the ``:sglang`` convenience aggregator
    and any future entry-point binaries live there and must survive
    regeneration.

    Sources come from the per-directory ``cycle_N_files`` filegroups that
    render_dir emits (a glob cannot cross package boundaries), so every cycle
    member is reachable as one collapsed target.
    """
    labels = [f"cycle_{i}" for i in range(len(g.sccs))]
    # Every module's provider target: the collapsed cycle target for cycle
    # members, its own per-directory target otherwise. This is the complete
    # sglang closure, so python/BUILD.bazel builds the :sglang aggregator from
    # it instead of re-deriving (and getting wrong, e.g. when a package's
    # __init__ is itself a cycle member and has no pkg target).
    all_deps = sorted({g.dep_label(m) for m in g.modules
                       if keep_dirs is None or g.pkg_dir(m) in keep_dirs})
    # Cycle targets gather sources only from directories that got BUILD files;
    # members outside the closure are absent from the Bazel graph entirely.
    lines = [GENERATED, "", '"""Generated import-cycle targets; loaded by python/BUILD.bazel."""', "",
             PY_LOAD, "",
             "# Labels of every collapsed import-cycle target, in stable order.",
             "CYCLE_LABELS = ["] + [f'    ":{l}",' for l in labels] + ["]", "",
             "# The provider target of every sglang module (the full package).",
             "ALL_MODULE_DEPS = ["] + [f'    "{d}",' for d in all_deps] + ["]", "",
             "def declare_cycles():",
             '    """One glob\'d py_library per import cycle; called from python/BUILD.bazel."""']
    if not g.sccs:
        lines.append("    pass")
    for i, members in enumerate(g.sccs):
        deps = set()
        srcs = set()
        data = set()
        in_closure = [m for m in sorted(members)
                      if keep_dirs is None or g.pkg_dir(m) in keep_dirs]
        if not in_closure:
            continue
        for m in in_closure:
            pkg = g.pkg_dir(m)
            srcs.add(f"//{g.src_root}/{pkg}:cycle_{i}_files")
            for dep in g.edges.get(m, ()):
                if g.droppable(dep):
                    continue  # NON_CLOSURE_DEPS; never reached on the TPU path
                if dep not in g.scc_of:
                    deps.add(g.dep_label(dep))
                elif g.scc_of[dep] != i:
                    deps.add(g.dep_label(dep))
            deps.update(g.ext_deps.get(m, ()))
        # Non-Python resource trees (include/, csrc/) a cycle member resolves
        # from the filesystem at import time -- possibly at an ancestor package
        # of the importing module (e.g. kernels/jit's KERNEL_PATH, found by
        # walking up from kernels/jit/utils/compile, a cycle-0 module). Attach
        # every resource tree directly under a package whose subtree this cycle
        # reaches into. Data, not deps: no PyInfo, no edge back into the cycle.
        for pkg in sorted(g.pkgs_with_resources):
            if keep_dirs is not None and pkg not in keep_dirs:
                continue
            if any(m.startswith(g.pkg_module_prefix[pkg]) for m in in_closure):
                for sub in g.pkgs_with_resources[pkg]:
                    data.add(f"//{g.src_root}/{pkg}:{sub}_data")
        lines += ["    py_library(", f'        name = "cycle_{i}",', "        srcs = ["]
        for s in sorted(srcs):
            lines.append(f'            "{s}",')
        lines += ["        ],", '        imports = ["."],', '        visibility = ["//visibility:public"],']
        if deps:
            lines += ["        deps = ["] + [f'            "{d}",' for d in sorted(deps)] + ["        ],"]
        if data:
            lines += ["        data = ["] + [f'            "{d}",' for d in sorted(data)] + ["        ],"]
        lines += ["    )", ""]
    return "\n".join(lines)


def write_if_changed(path: Path, content: str, check: bool):
    if path.exists() and path.read_text() == content:
        return
    if check:
        raise SystemExit(f"{path} is stale; run bazel run //bazel:gen_sglang_build")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="python", help="dir containing the sglang package")
    ap.add_argument(
        "--external",
        default=None,
        help="bazel output_base/external dir holding the materialized pip hub; "
             "default: resolved from `bazel info output_base`",
    )
    ap.add_argument("--hub", default="sglang_pypi", help="pip.parse hub_name")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any generated file would change")
    args = ap.parse_args()

    src_root = Path(args.src)
    external = args.external
    if external is None:
        import subprocess
        external = subprocess.run(
            ["bazel", "info", "output_base"], capture_output=True, text=True, check=True
        ).stdout.strip() + "/external"
    hub_modules = hub_top_level_modules(Path(external), args.hub)
    g = Graph(src_root, args.hub, hub_modules)

    mods_by_dir = defaultdict(list)
    for mod in g.modules:
        mods_by_dir[g.pkg_dir(mod)].append(mod)
    keep_dirs = closure_dirs(g)
    g.keep_dirs = keep_dirs
    leaks = {
        (m, d)
        for m in g.modules
        if g.pkg_dir(m) in keep_dirs
        for d in g.edges.get(m, ())
        if g.pkg_dir(d) not in keep_dirs and not _is_non_closure_allowed(d)
    }
    if leaks:
        detail = "\n".join(f"  {m} -> {d}" for m, d in sorted(leaks))
        raise SystemExit(
            "closure modules depend on directories outside the closure "
            "(add the root to DYNAMIC_ROOTS or the module to NON_CLOSURE_DEPS):\n"
            + detail
        )
    stale = 0
    for pkg_dir, mods in sorted(mods_by_dir.items()):
        build = src_root / pkg_dir / "BUILD.bazel"
        if pkg_dir not in keep_dirs:
            # Outside the TPU closure: no Bazel footprint. Remove a BUILD file
            # an earlier (less selective) generator run left behind.
            if build.exists():
                if args.check:
                    raise SystemExit(f"{build} is stale; run bazel run //bazel:gen_sglang_build")
                build.unlink()
                stale += 1
                print(f"removed {build} (outside TPU closure)")
            continue
        write_if_changed(build, render_dir(g, pkg_dir, mods), args.check)

    # One glob'd py_library per import cycle, living at the python/ root so it
    # may gather files from every directory the cycle spans. Emitted as a .bzl
    # macro that the handwritten python/BUILD.bazel loads, so that file can
    # also carry the handwritten :sglang aggregator without being clobbered.
    write_if_changed(src_root / "cycles.bzl", render_cycles(g, keep_dirs), args.check)

    report = {
        "modules": len(g.modules),
        "dirs_emitted": len(keep_dirs),
        "dirs_total": len(mods_by_dir),
        "dirs_pruned": stale,
        "sccs": [{"size": len(s), "anchor": min(s, key=lambda m: (m.count("."), m))}
                 for s in g.sccs],
        "unknown_top_level_imports": {k: sorted(v) for k, v in sorted(g.unknown.items())},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
