"""Drop checkpoint files from the page cache: python3 drop_weight_cache.py <dir> [...].

After load the weights live in TPU HBM, but the host keeps their pages as "active" file
cache, which counts toward the pod's working set (what memguard and eviction watch).
posix_fadvise(DONTNEED) releases them; pages still mapped by a process are kept.
"""
import os, sys

freed = 0
for root in sys.argv[1:]:
    for dirpath, _, names in os.walk(root, followlinks=True):
        for name in names:
            path = os.path.join(dirpath, name)
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                freed += os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
print(f"fadvised {freed / 2**30:.1f} GiB of files")
