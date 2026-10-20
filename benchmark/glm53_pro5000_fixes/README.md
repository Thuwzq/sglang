# GLM-5.3-Flash PRO5000 serving fixes

This branch carries the deployed SGLang v0.5.20 fixes for the eight-card
RTX PRO 5000 SM120 setup. It already included the smaller TileLang NoPE tile
and KDA host-length reuse. This update adds the following serving repairs.

| Repair | Trigger and resulting behavior |
| --- | --- |
| Hybrid HiCache DSA index restore | Register INDEXER alongside KV/MAMBA, use the actual KV row width, and restore the target/draft index state with its corresponding KV pages. |
| Packed MTP row guard | Require matching MLA storage widths and dtypes; use the existing single-draft sidecar path for supported mismatches and reject unsupported recurrent-state combinations. |
| Recurrent checkpoint ownership | Do not cache a finished request's recurrent snapshot under a shorter prefix key. |
| Indexer host-buffer cleanup | Unregister the separately allocated pinned index buffer and make destruction idempotent. |
| Spec-v2 grammar barrier | Before target prefill, including mixed extend, process the pending grammar result before constructing the next constrained sampling state. |
| Adaptive Graph pruning | Synchronize the legacy batch override with the resolved decode Graph configuration; disable empty captures and restore both configurations on exit. |

The Graph fix is in `EAGLEWorkerV2._override_worker_state()`. Runners consume
`cuda_graph_config.decode`; updating only `cuda_graph_bs_decode` left redundant
captures enabled. A deep copy now carries the temporary batch list, maximum
batch and disabled backend where applicable, with restoration in `finally`.

## Reproducing CPU regressions

Use Python 3.10 or newer (the production sources contain match statements):

```bash
python3 test/manual/spec/test_glm53_pro5000_fix_contracts.py -v
```

The standard-library tests execute extracted production methods with stubbed
dependencies. They cover eight Graph override/restore scenarios (including
empty sets and exceptions), constrained/plain extend and mixed-extend order,
packed-row geometry/dtype checks, and the checkpoint overshoot rejection.
They do not validate CUDA kernels, byte transfers, distributed races, or
full-model numerical accuracy. These local CPU checks were rerun for this
update; a fresh GPU benchmark was not run.

## Previous device validation

The following are historical PRO5000 observations, not CI guarantees:

- HiCache GPU-to-host-to-relocated-GPU checks covered KV/index rows of widths
  512 and 576, plus recurrent/conv state; natural host reuse and changed-fact
  multi-turn requests passed in the recorded test scope.
- Grammar GPU A/B: the original configuration completed 37/43 protocol
  requests; disabling mixed chunk or applying the barrier completed 43/43
  each. The deployed barrier subsequently passed another 43 structured and
  8 plain protocol requests. Protocol completion is not a full accuracy gate.
- Graph32 deployment passed 32/32 short streams, observed an actual batch32
  Graph hit, and passed a structured probe with sufficient output budget.
  The original deployment record also reported eight CPU override cases.
- The TP0 startup capture window consumed 7.47 GB before the Graph fix and
  3.55 GB after it. This includes inter-capture initialization. It is not
  isolated Graph allocation or measured steady-state savings on every rank.
  No Graph16 startup with the same fix was measured, so there is no controlled
  pure Graph16-to-32 memory or throughput delta.
- Later bounded monitoring reviewed 127,055 untruncated log lines through
  2026-10-20 09:48 Asia/Shanghai, with 16 successful health checks and no
  matching grammar/OOM/allocator-retry/CUDA/NCCL recurrence. That does not
  establish 100% external request success or full context/concurrency coverage.

Four production source files match the local deployment snapshots byte for
byte. `base_spec_worker.py` has formatting-only line wrapping; its parsed AST
is identical to the deployed snapshot (SHA-256
`1378c2df98978472ee7005b1de6441f37ee7454887c5c5ca1be39ae5f0ca210a`).
The repository source SHA-256 values are:

```text
f55058ce733d546fd0db038a3b3b2f50eb3187132372bbe73cfcb2dcdd011251  python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
c9c36b655d7cc13724c26a60a98e53242d5b6dc7e981f2d8a48743528651c21b  python/sglang/srt/mem_cache/pool_host/dsa.py
24dcf9a05769d1314a2a4f11dc27b68395b220c3e7f4a0be915b9d950f7f79ba  python/sglang/srt/mem_cache/unified_cache/components/mamba.py
f9cf0058beb2ab65c9743856f712fcd3f8973cc58b2dc1ad259e79394d44f616  python/sglang/srt/speculative/base_spec_worker.py
0fad0647e2aac542ab6390805bdb2bef22f44959e3f094d546eaf4c3a618287a  python/sglang/srt/speculative/eagle_worker_v2.py
```

## Configuration and limits

The last deployed profile used native FP8 weights, BF16 DSA KV, TP8/EP1,
attention input-scattered SP, adaptive native MTP, mixed chunk, decode Graph
buckets 1/2/4/8/16/32, prefill chunk/token budget 1024, a shared device token
capacity of 1,064,960, 192 recurrent-cache slots, and fixed host-cache budget
180 GB per rank (about 1.41 TiB total). This is a hardware-specific capacity
profile, not a portable default; eight concurrent 1M requests were not qualified.
It does not implement FlashInfer native NoPE FP8 KV storage.

Historical SSE disconnects, an observed abort counter of 11, and driver-visible
GPU headroom falling to 4–8 MiB before recovering remain unassigned symptoms.
This update does not claim to fix them. Likewise, no bitwise model equivalence
or full 1M capacity matrix is established by these regressions.

The user stopped the serving container on 2026-10-20. Publishing this source
update does not restart the service or resume log monitoring. Existing launch
scripts and deployment records should be checked before an intentional restart.
