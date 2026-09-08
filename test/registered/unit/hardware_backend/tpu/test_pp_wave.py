"""Unit tests for the TPU pipeline hand-off wave.

CPU only: the collective-permute is a threaded fake that pairs launch k on
every stage and hands each stage's payload to the next, blocking a stage until
its peers reach the same launch, the way the device does.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import threading
import unittest

import torch

from sglang.srt.hardware_backend.tpu.pp_wave import PPWave
from sglang.test.test_utils import CustomTestCase

ROWS, HIDDEN = 8, 4


class _FakeIci:
    """Launch k on every stage exchanges payloads: stage r receives what stage
    r-1 sent at launch k (zeros on stage 0)."""

    def __init__(self, size: int, timeout: float = 10.0):
        self.size = size
        self.timeout = timeout
        self.cond = threading.Condition()
        self.sent: dict[tuple[int, int], tuple] = {}
        self.count = [0] * size
        self.failed = False

    def launcher(self, rank: int):
        def launch(h, r):
            with self.cond:
                k = self.count[rank]
                self.sent[(k, rank)] = (h.clone(), r.clone())
                self.count[rank] += 1
                self.cond.notify_all()
                ok = self.cond.wait_for(
                    lambda: self.failed or all(c > k for c in self.count),
                    timeout=self.timeout,
                )
                if not ok or self.failed:
                    self.failed = True
                    self.cond.notify_all()
                    raise RuntimeError(
                        f"stage {rank} launch {k}: peers never launched it "
                        f"(counts {self.count})"
                    )
                if rank == 0:
                    return tuple(torch.zeros_like(t) for t in (h, r))
                return self.sent[(k, rank - 1)]

        return launch


def _wave(rank, size, ici=None):
    return PPWave(
        rank=rank,
        size=size,
        rows=ROWS,
        hidden=HIDDEN,
        dtype=torch.float32,
        device=torch.device("cpu"),
        launcher=ici.launcher(rank) if ici else (lambda h, r: (h, r)),
    )


def _expected_counts(size, schedule):
    """Per iteration, the launch count every stage must have reached: stage 0
    launches once per iteration with a batch or while it owes the drain of
    2 * (size - 2), stage r keeps min(r-1, owed) launches ahead of it."""
    counts = []
    stage0, owed = 0, 0
    for has_batch in schedule:
        if has_batch:
            stage0, owed = stage0 + 1, 2 * (size - 2)
        elif owed:
            stage0, owed = stage0 + 1, owed - 1
        counts.append([stage0] + [stage0 + min(r - 1, owed) for r in range(1, size)])
    return counts


def _run_schedule(size, schedule, rows_per_batch=None):
    """Every stage runs ``schedule`` (True = an iteration with a batch). Stage 0
    emits batch j as the value 100 + j; each later stage adds one. Returns the
    per-stage received values, the per-stage launch count after every
    iteration, the waves, any stage error and the fake ICI."""
    ici = _FakeIci(size)
    waves = [_wave(r, size, ici) for r in range(size)]
    received = [[] for _ in range(size)]
    counts = [[] for _ in range(size)]
    errors = []

    def run(rank):
        wave = waves[rank]
        batch = 0
        try:
            for has_batch in schedule:
                wave.begin_iteration(has_batch)
                tensors = None
                if has_batch:
                    rows = rows_per_batch[batch] if rows_per_batch else ROWS
                    incoming = wave.receive()
                    if rank == 0:
                        out = torch.full((rows, HIDDEN), 100.0 + batch)
                    else:
                        received[rank].append(float(incoming["hidden_states"][0, 0]))
                        self_check = float(incoming["residual"][rows - 1, HIDDEN - 1])
                        assert self_check == received[rank][-1] + 0.5, (rank, batch)
                        out = incoming["hidden_states"][:rows] + 1.0
                    if rank < size - 1:
                        tensors = {"hidden_states": out, "residual": out + 0.5}
                    batch += 1
                wave.end_iteration(tensors)
                counts[rank].append(wave.launches)
        except Exception as exc:  # noqa: BLE001
            errors.append((rank, exc))
            with ici.cond:
                ici.failed = True
                ici.cond.notify_all()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return received, counts, waves, errors, ici


def _drained(size, schedule):
    return schedule + [False] * (2 * (size - 2))


class TestWaveProtocol(CustomTestCase):
    def test_activations_reach_every_stage_in_order_across_gaps(self):
        """Batch j leaves stage 0 as 100+j and must arrive at stage r as
        100+j+r-1 for every j: across gaps inside the drain, a gap longer
        than the drain, and single batches."""
        size = 4
        schedule = _drained(
            size,
            [True, True, True, False, False, True, False, True, True]
            + [False] * 5
            + [True],
        )
        received, counts, waves, errors, ici = _run_schedule(size, schedule)
        self.assertEqual(errors, [])
        n = schedule.count(True)
        for rank in range(1, size):
            self.assertEqual(
                received[rank], [100.0 + j + rank - 1 for j in range(n)], rank
            )
        self.assertEqual(len(set(ici.count)), 1)
        for wave in waves:
            self.assertIsNone(wave.pending)
            self.assertEqual(wave.owed, 0)
            self.assertEqual(wave.launches, wave.stage0_launches)

    def test_stage_zero_launches_one_hop_per_iteration_and_stages_keep_pace(self):
        """Stage 0 issues at most one launch per iteration (a send, or a zero
        launch while draining), and stage r is at most r-1 launches ahead."""
        size = 5
        schedule = _drained(size, [True] * 3 + [False] * 7 + [True])
        _, counts, _, errors, _ = _run_schedule(size, schedule)
        self.assertEqual(errors, [])
        expected = _expected_counts(size, schedule)
        for rank in range(size):
            self.assertEqual(
                counts[rank], [row[rank] for row in expected], f"stage {rank}"
            )
        # A gap longer than the drain stops every stage; the run restarts
        # with stage r launching r-1 hops ahead of stage 0 again.
        self.assertEqual(expected[9], [9, 9, 9, 9, 9])
        self.assertEqual(expected[10], [10, 10, 11, 12, 13])

    def test_drain_is_twice_stages_minus_two_launches_after_the_last_batch(self):
        size = 5
        schedule = _drained(size, [True] * 6 + [False])
        _, _, waves, errors, ici = _run_schedule(size, schedule)
        self.assertEqual(errors, [])
        self.assertEqual(ici.count, [6 + 2 * (size - 2)] * size)
        for wave in waves:
            self.assertEqual(wave.owed, 0)
            self.assertIsNone(wave.pending)

    def test_smaller_batches_are_padded_to_the_buffer_and_trimmed_by_rows(self):
        size = 3
        schedule = _drained(size, [True, True, False])
        received, _, _, errors, _ = _run_schedule(size, schedule, rows_per_batch=[3, 5])
        self.assertEqual(errors, [])
        self.assertEqual(received[2], [101.0, 102.0])

    def test_two_stages_launch_once_per_batch_and_nothing_else(self):
        size = 2
        schedule = [True, True, False, True, False]
        received, _, _, errors, ici = _run_schedule(size, schedule)
        self.assertEqual(errors, [])
        self.assertEqual(received[1], [100.0, 101.0, 102.0])
        self.assertEqual(ici.count, [3, 3])


class TestWaveGuards(CustomTestCase):
    def test_rejects_rows_above_the_buffer(self):
        wave = _wave(0, 2)
        wave.begin_iteration(True)
        big = torch.zeros(ROWS + 1, HIDDEN)
        with self.assertRaisesRegex(ValueError, "above the wave"):
            wave.end_iteration({"hidden_states": big, "residual": big})

    def test_receive_needs_an_iteration_with_a_batch(self):
        wave = _wave(1, 3)
        wave.begin_iteration(False)
        with self.assertRaisesRegex(RuntimeError, "outside an iteration"):
            wave.receive()

    def test_a_second_output_without_a_launch_between_is_an_error(self):
        wave = _wave(1, 3)
        out = {
            "hidden_states": torch.zeros(ROWS, HIDDEN),
            "residual": torch.zeros(ROWS, HIDDEN),
        }
        wave.begin_iteration(True)
        wave.receive()
        wave.end_iteration(out)
        with self.assertRaisesRegex(RuntimeError, "two forward outputs"):
            wave.end_iteration(out)

    def test_probe_checks_the_received_stage_number(self):
        size = 3
        ici = _FakeIci(size)
        waves = [_wave(r, size, ici) for r in range(size)]
        verdicts = [None] * size

        def run(rank):
            verdicts[rank] = waves[rank].probe()

        threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(verdicts, [True] * size)
        self.assertEqual(len(set(ici.count)), 1)

        def wrong(h, r):
            return (torch.full_like(h, 1.0), torch.full_like(r, 1.0))

        wave = _wave(2, 3)
        wave._launcher = wrong
        self.assertFalse(wave.probe())


if __name__ == "__main__":
    unittest.main()
