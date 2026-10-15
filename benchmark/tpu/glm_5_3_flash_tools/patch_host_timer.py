"""Insert (or with --undo remove) the QH_HOST_TIMER hook in the pod's scheduler.py."""
import sys

P = "/dev/shm/home/qinghanc/work/glm-dsa-tpu/python/sglang/srt/managers/scheduler.py"
ANCHOR = '        """A scheduler loop that overlaps the CPU processing and GPU computation."""\n'
HOOK = (
    '        if __import__("os").environ.get("QH_HOST_TIMER"):  # QH_HOST_TIMER\n'
    "            from sglang.srt.managers import qh_host_timer; qh_host_timer.install(self)  # QH_HOST_TIMER\n"
)
s = open(P).read()
if "--undo" in sys.argv:
    s = "".join(l for l in s.splitlines(True) if "# QH_HOST_TIMER" not in l)
elif HOOK not in s:
    assert s.count(ANCHOR) == 1
    s = s.replace(ANCHOR, ANCHOR + HOOK)
open(P, "w").write(s)
print("hooked" if HOOK in s else "clean")
