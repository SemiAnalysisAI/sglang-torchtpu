# Results: gpt-oss-120b on one tpu7x-8 host, 2026-10-20

Six one-hour runs, in parallel on six tpu7x-8 hosts, with the setup in [REPRODUCE.md](REPRODUCE.md):
arms A-D, plus a repeat of C and of D (C2, D2) to show the run-to-run spread. Arm E (both #1586
commits, the copy-ahead module exactly as at `78f109fc3f`) ran later the same day, in both modes, on
the same kind of host. The copy-ahead module ran
the same code as the commits named in REPRODUCE.md (plus a test-only switch selecting the merged pin
for C and the fixed pin for D); other cache code differed only in features that
are off by default, and other parts of the stack differed, so absolute throughput on another build can
differ. All arms used the fixed-date chat template, the client at its
defaults (20 sessions, seed 1001), and `SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM=1` for the
write_back arms. Table from `compare.py`.

| Arm | Requests | Req/s | Hit % | Device / host / new % | Mean sampled decode tok/s | Main latency p50 / p90 / p99 s | Eviction s (TP0) |
|---|---|---|---|---|---|---|---|
| A write_through | 4,973 | 1.381 | 68.9 | 3.1 / 65.8 / 31.1 | 341 | 29.2 / 46.9 / 59.1 | 25 |
| B write_back, window off | 4,797 | 1.333 | 73.1 | 3.2 / 69.9 / 26.9 | 320 | 30.7 / 49.2 / 60.6 | 1,880 |
| C write_back, window on (main) | 2,235 | 0.621 | 10.7 | 5.9 / 4.8 / 89.3 | 223 | 73.2 / 105.2 / 128.1 | 1,697 |
| C2 (repeat of C) | 2,217 | 0.616 | 10.6 | 6.4 / 4.2 / 89.4 | 245 | 67.2 / 101.4 / 124.0 | 1,674 |
| D write_back, window on, fix | 5,497 | 1.527 | 71.2 | 3.0 / 68.2 / 28.8 | 335 | 29.1 / 41.9 / 56.7 | 732 |
| D2 (repeat of D) | 5,112 | 1.420 | 72.9 | 2.9 / 70.0 / 27.1 | 282 | 31.7 / 45.3 / 61.3 | 600 |
| E write_back, window on, both #1586 commits | 4,970 | 1.381 | 72.4 | 3.1 / 69.2 / 27.6 | 364 | 29.8 / 47.3 / 59.5 | 853 |

Hit % per 10 minutes:

| Minutes | A | B | C | C2 | D | D2 |
|---|---|---|---|---|---|---|
| 0-10 | 69.9 | 68.7 | 9.1 | 8.9 | 69.8 | 70.0 |
| 10-20 | 76.4 | 78.7 | 9.1 | 9.4 | 76.1 | 76.9 |
| 20-30 | 66.8 | 70.1 | 11.9 | 11.0 | 66.6 | 71.5 |
| 30-40 | 65.5 | 77.8 | 9.9 | 9.9 | 74.8 | 77.9 |
| 40-50 | 63.3 | 70.1 | 9.8 | 10.9 | 70.3 | 68.4 |
| 50-60 | 68.8 | 72.1 | 14.5 | 13.3 | 68.4 | 71.6 |

HiCache counters, TP rank 0, millions of tokens:

| | A | B | C | C2 | D | D2 |
|---|---|---|---|---|---|---|
| Backed up to host, full / SWA pool | 28.0 / 37.9 | 26.7 / 34.5 | 50.8 / 57.3 | 50.5 / 57.4 | 30.8 / 40.6 | 28.7 / 36.9 |
| Loaded back from host, full / SWA pool | 116.7 / 5.8 | 120.5 / 5.8 | 2.1 / 1.8 | 1.9 / 1.7 | 134.1 / 6.4 | 128.2 / 6.1 |
| Retracted requests | 1 | 2 | 3 | 5 | 3 | 4 |

Client: 0 errors and no worker crashes in every arm. 32-49 requests per arm were still running at the
hour and finished within 44 s. 97% of replies ended at `max_tokens`.

- **The window as merged (C) loses almost all host hits.** Its hit rate is 9-15% in every 10-minute
  bucket, against 69-79% for B. It backs up twice as much KV to host as B, but loads back 2M tokens
  against 120M. It serves 53% fewer requests than B.
- **The fix (D) restores B's hit rate** (71.2% and 72.9% against 73.1%) and keeps the window's
  benefit: eviction time falls from 1,880 s to 600-732 s, and D serves 7-15% more requests than B.
- **E (both #1586 commits) also restores B's hit rate** (72.4%, prefill-only 71.8%) with eviction time
  853 s (prefill-only 1,186 s against 1,839 s for PD). Its requests (4,970) fall inside D's
  run-to-run spread, so the second commit gives no measurable throughput change on this load.
- The repeats agree: C and C2 within 1% on requests, D and D2 within about 7-8%.
- write_through (A) and write_back without the window (B) are close on this load: B has a higher hit
  rate and serves 4% fewer requests.
- The first ~2 minutes of each run are cold: every session's first request has a long uncached prompt
  (B: 10.8% hit in the first 120 s, 82% by minutes 5-10).

## Prefill-only variant

The same four arms with the client's `--prefill-only` (one output token per request), one run each.

| Arm | Requests | Hit % | Input tok/s | Uncached input tok/s | Main TTFT p50 / p90 / p99 s | Reminder TTFT p50 / p90 / p99 s | Eviction s (TP0) |
|---|---|---|---|---|---|---|---|
| PA write_through | 5,877 | 67.9 | 63,404 | 20,407 | 21.6 / 39.1 / 72.2 | 22.4 / 41.3 / 75.5 | 30 |
| PB write_back, window off | 6,202 | 72.3 | 67,081 | 18,645 | 20.7 / 35.8 / 71.0 | 21.3 / 39.2 / 78.8 | 2,716 |
| PC write_back, window on (main) | 2,131 | 9.5 | 23,211 | 21,000 | 68.0 / 95.4 / 108.3 | 69.9 / 96.7 / 112.7 | 3,187 |
| PD write_back, window on, fix | 5,797 | 72.0 | 62,502 | 17,538 | 23.0 / 36.7 / 69.3 | 23.6 / 39.3 / 75.8 | 1,839 |
| PE write_back, window on, both #1586 commits | 5,651 | 71.8 | 60,823 | 17,171 | 22.4 / 38.9 / 69.6 | 24.1 / 41.7 / 75.2 | 1,186 |

| Minutes | PA | PB | PC | PD |
|---|---|---|---|---|
| 0-10 | 10.3 | 10.4 | 10.2 | 10.3 |
| 10-20 | 61.6 | 76.9 | 9.0 | 76.1 |
| 20-30 | 77.6 | 78.2 | 10.5 | 77.7 |
| 30-40 | 75.5 | 79.1 | 9.0 | 79.2 |
| 40-50 | 74.2 | 79.1 | 8.3 | 78.2 |
| 50-60 | 73.9 | 70.1 | 9.6 | 74.8 |

- PC loses host hits as C does: it backs up 71M full-pool tokens and loads back 0.7M (PB: 33M and 154M).
- PD restores PB's hit rate (72.0% against 72.3%) and serves 93% of PB's requests, within the
  about 7-8% run-to-run spread of the normal-mode repeats; its eviction time is 68% of PB's.
- The first 10 minutes are nearly cold in every arm (about 10%), over 445-513 requests (in PB, a mean
  prompt of 39k tokens, and only 314 requests in minutes 2-10, which looks like a server stall rather
  than a cold cache). Why this window is colder than in the normal run was not investigated.
- 0 errors and no worker crashes in every arm.

## Prompt sizes

Measured prompt sizes (arm D): a turn-0 main prompt is 7,660 tokens (system prompt about 7.2k plus the
task); turn-0 reminder prompts are 3.7-6.2k tokens (reminder system prompt, task and first tool result);
each turn adds 2.5k tokens (median, range 1.3-3.8k); the largest main prompt is 75k tokens. Initial
conversations start at turns 1-20, so most first requests are longer than turn 0. The decode column is
the mean of the scheduler's sampled `gen throughput` values.
