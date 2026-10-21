# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Geometry and tiling of the GPT-OSS TP8 decode megakernel (see DESIGN.md)."""

from __future__ import annotations

import dataclasses

BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
# Timing ablations (outputs become wrong): "ar" keeps pack + local reduce but skips the
# cross-rank exchange (and the start barrier); "attn" skips the whole attention block incl.
# its all-reduce; "attn_kv" skips only the paged attention core; "moe" skips router, experts
# and their all-reduce; "experts" skips only the expert matmuls/stream.
ABLATIONS = ("ar", "attn", "attn_kv", "moe", "experts")


def rup(x: int, m: int = 128) -> int:
    return (x + m - 1) // m * m


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclasses.dataclass(frozen=True)
class DecodeConfig:
    """One decode step (T=1 token per sequence) of B sequences through all L layers at TP=R.

    Layer l is sliding-window when `l % 2 == 0` (GPT-OSS `layer_types`), so L must be even;
    sliding layers read the SWA KV pool, full layers the full KV pool (sglang's SWAKVPool).
    """

    B: int = 1  # batch bucket (sequences per step, static per compile)
    Q: int = 1  # tokens per sequence (1 for DECODE, VerifyWidth for TARGET_VERIFY)
    L: int = 36
    H: int = 2880
    NQ: int = 64
    NKV: int = 8
    D: int = 64
    E: int = 128
    TOPK: int = 4
    I: int = 2880
    window: int = (
        128  # attended tokens including the current one (sglang window_left 127)
    )
    eps: float = 1e-5
    alpha: float = 1.702
    limit: float = 7.0
    rope_theta: float = 150000.0
    rope_factor: float = 32.0
    rope_orig: int = 4096
    R: int = 8  # TP ranks
    moe_mode: str = (
        "tp"  # "tp": every rank holds I/R of every expert; "ep": E/R whole experts
    )
    kv_dtype: str = (
        "bf16"  # "bf16" or "fp8" (float8_e4m3fn), sglang TPU fused-KV layout
    )
    page: int = 64
    max_pages: int = (
        128  # page-table columns per sequence (context <= page * max_pages)
    )
    nb_full: int = 1024  # pages in the full-attention pool (per rank, per layer)
    nb_swa: int = 1024  # pages in the SWA pool
    npb: int = 8  # KV pages per attention work item, full layers (one DMA per page)
    npb_swa: int = 3  # pages per item, sliding layers (>= NWP: one item per sequence)
    kv_bufs: int = 3  # KV item buffers (kv_bufs - 1 items prefetched ahead of compute)
    tne: int = 384  # expert intermediate chunk (lanes) per streamed work item
    w_slots: int = 3  # expert weight ring depth (w_slots - 1 items in flight)
    vmem_mib: int = 60
    ar_dtype: str = "bf16"  # all-reduce payload dtype (sum is always f32)
    # expert weights: "bf16"; "mxfp4" (e2m1 codes + scale per 32 K, dequantized in VMEM);
    # "fp8" (MXFP4 converted offline to e4m3 x per-column power of two: exact, native MXU)
    w_dtype: str = "fp8"
    ar_fold: bool = (
        True  # BQ <= 4: all-reduce only the live rows, folded to [BQ*Hp/128, 128] f32
    )
    # "oneshot": push the full partial to all peers (1 round, 7x payload per rank);
    # "rsag": reduce-scatter + all-gather of Hp/R column chunks (2 rounds, ~2.5x fewer bytes);
    # "auto": rsag for BQ >= 16 (bandwidth-bound), oneshot below (latency-bound)
    ar_mode: str = "auto"
    ablate: tuple[str, ...] = ()  # see ABLATIONS
    # "scatter": new K/V written by a separate non-collective call (safe default);
    # "kernel": the megakernel writes them into the aliased pools itself (one call per step)
    kv_write: str = "scatter"
    # KV pools as L/2 SWA + L/2 full per-layer operands (sglang keeps one tensor per layer;
    # stacking them would copy the pools) instead of two stacked [L/2, ...] operands
    split_pools: bool = False
    # Layer boundary indices to capture into an auxiliary HBM output (EAGLE3 / DFlash):
    # index `b` captures the residual stream before layer `b` (i.e. after layer `b - 1`).
    capture_layers: tuple[int, ...] = ()
    # Decode context parallelism (DCP): a group of `dcp_size` ranks in the TP mesh splits
    # each sequence's KV positions by `pos % dcp_size == dcp_rank` and merges partial
    # softmax accumulators in VMEM via remote DMA.
    dcp_size: int = 1
    # Whether SWA layers also use DCP-sharded KV pools (True for SGLang --dcp-size serving
    # where SWAKVPool shards both pools; False when SWA pools stay unsharded DCP=1).
    dcp_swa: bool = False
    # When True at dcp_size=4 and kv_dtype="fp8", packs 2 KV heads into lanes 0..63 and
    # 64..127 of each fp8 packing slot so all 4 KV heads fit in kv_groups=1 ([page, 1, 4, 128]).
    dcp_lane_pack: bool = False

    def __post_init__(self) -> None:
        # At BR >= 256 (e.g. B=64, Q=4), RSAG + activation buffers consume ~18 MiB more VMEM
        # than at BR=64; double-buffering expert weights (w_slots=2) and KV blocks (kv_bufs=2)
        # saves 3.71 MiB and keeps total scoped VMEM at 56.3 MiB (< 60.0 MiB TPU v7x limit).
        if self.BR >= 256 and self.H >= 2880:
            if self.w_slots > 2:
                object.__setattr__(self, "w_slots", 2)
            if self.kv_bufs > 2:
                object.__setattr__(self, "kv_bufs", 2)

    # ------------------------------------------------------------------ derived
    @property
    def BQ(self) -> int:  # total tokens across B sequences
        return self.B * self.Q

    @property
    def Hp(self) -> int:
        return rup(self.H)

    @property
    def BR(self) -> int:  # activation rows (bf16 sublane tile)
        return rup(self.BQ, 16)

    @property
    def NQL(self) -> int:  # query heads per rank
        return self.NQ // self.R

    @property
    def QWL(self) -> int:
        return self.NQL * self.D

    @property
    def QKVWL(self) -> int:  # q | k (1 head, padded to 128 lanes) | v (same)
        return self.QWL + 256

    @property
    def EL(self) -> int:  # experts per rank
        return self.E if self.moe_mode == "tp" else self.E // self.R

    def moe_cols(self, r: int) -> tuple[int, int]:
        """Rank r's slice [start, stop) of the expert intermediate dim.

        "tp" + bf16: even I/R split. "tp" + mxfp4/fp8: whole 32-wide scale blocks per rank
        (90 blocks over 8 ranks -> 12,12,11,...), so no block straddles two ranks.
        """
        if self.moe_mode == "ep":
            return 0, self.I
        if self.w_dtype == "bf16":
            return r * (self.I // self.R), (r + 1) * (self.I // self.R)
        nb = self.I // 32
        base, extra = divmod(nb, self.R)
        start = r * base + min(r, extra)
        return 32 * start, 32 * (start + base + (r < extra))

    @property
    def IL_RAW(self) -> int:  # largest per-rank intermediate slice
        return max(b - a for a, b in (self.moe_cols(r) for r in range(self.R)))

    @property
    def IL(self) -> int:
        return rup(self.IL_RAW, self.tne)

    @property
    def NJ(self) -> int:
        return self.IL // self.tne

    @property
    def kv_packing(self) -> int:  # KV elements per 32-bit word (sglang `kv_packing`)
        return 4 if self.kv_dtype == "fp8" else 2

    @property
    def kv_groups(self) -> int:  # axis-2 of fused pool [num_blocks, page, kv_groups, packing, 128]
        if self.dcp_size == 1:
            return 1
        if self.dcp_lane_pack and self.kv_dtype == "fp8":
            return max(1, self.dcp_size // 4)
        return max(1, (2 * self.dcp_size) // self.kv_packing)

    @property
    def NWP(self) -> int:  # max pages holding the window's history tokens
        return cdiv(self.window - 1, self.page) + 1

    @property
    def TKMAX(self) -> int:  # tokens per KV item buffer
        return max(self.npb, self.npb_swa) * self.page

    @property
    def NIMAX(self) -> int:  # attention work items per layer, upper bound
        return self.B * max(cdiv(self.max_pages, self.npb), 1) + self.B

    @property
    def SR13(
        self,
    ) -> int:  # MXFP4 scale rows of a w13 chunk (Hp/32, padded to the bf16 tile)
        return rup(self.Hp // 32, 16)

    @property
    def SR2(self) -> int:  # MXFP4 scale rows of a w2 chunk
        return rup(self.tne // 32, 16)

    @property
    def AR_RSAG(self) -> bool:
        return self.R > 1 and (
            self.ar_mode == "rsag" or (self.ar_mode == "auto" and self.BQ >= 16)
        )

    @property
    def AR_FOLD(self) -> bool:
        return self.ar_fold and self.BQ <= 4 and not self.AR_RSAG

    @property
    def CWA(
        self,
    ) -> int:  # rsag column chunk per rank (lane multiple; last chunks padded)
        return rup(cdiv(self.Hp, self.R), 128)

    @property
    def FR(self) -> int:  # folded all-reduce payload rows
        return rup(self.BQ * (self.Hp // 128), 8)

    @property
    def VW(self) -> int:
        return max(self.Hp, self.QKVWL, 128)

    @property
    def sliding(self) -> tuple[bool, ...]:
        return tuple(l % 2 == 0 for l in range(self.L))

    @property
    def sm_scale(self) -> float:
        return self.D**-0.5

    def check(self) -> None:
        assert self.B >= 1 and self.Q >= 1
        assert self.D == 64, "head-pair lane packing assumes head_dim 64"
        assert self.NKV == self.R, "v0: exactly one KV head per rank"
        assert self.dcp_size in (1, 2, 4) and self.R % self.dcp_size == 0
        if self.dcp_size > 1:
            assert self.Q == 1, "DCP currently supports plain decode (Q=1)"
        assert self.NQL % 2 == 0 and self.QWL % 128 == 0
        assert self.L % 2 == 0, "layers come in (sliding, full) pairs"
        assert self.E <= 128 and self.TOPK <= self.E and self.E % self.R == 0
        assert self.I % self.R == 0 and self.moe_mode in ("tp", "ep")
        assert self.kv_dtype in ("bf16", "fp8") and self.ar_dtype in ("bf16", "f32")
        assert self.tne % 128 == 0 and self.w_slots >= 2
        assert self.w_dtype in ("bf16", "mxfp4", "fp8")
        assert set(self.ablate) <= set(ABLATIONS), self.ablate
        assert self.kv_write in ("scatter", "kernel")
        assert self.ar_mode in ("auto", "oneshot", "rsag")
        if self.w_dtype != "bf16":
            assert self.I % 32 == 0 and self.Hp % 128 == 0
        assert self.page % 8 == 0 and self.kv_bufs >= 2
        assert self.npb_swa >= self.NWP, "sliding layers: one work item per sequence"
        assert all(0 <= l <= self.L for l in self.capture_layers), self.capture_layers


FULL = DecodeConfig()


def full(B: int, **kw) -> DecodeConfig:
    return dataclasses.replace(FULL, B=B, **kw)


def tiny(B: int = 4, **kw) -> DecodeConfig:
    """All structural constraints of the real config at a size the CPU interpreter runs.

    H=240 exercises the hidden padding (Hp=256); NQ/NKV/D are the real head geometry so the
    attention code path is exactly the real one; E=16 < 128 exercises the router padding.
    """
    base = dict(
        L=2,
        H=240,
        E=16,
        TOPK=4,
        I=8 * 48,
        max_pages=4,
        nb_full=24,
        nb_swa=24,
        npb=2,
        npb_swa=3,
        kv_bufs=3,
        tne=128,
        w_slots=3,
        vmem_mib=32,
    )
    base.update(kw)
    return DecodeConfig(B=B, **base)
