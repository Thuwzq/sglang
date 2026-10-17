# GLM-5.3-Flash KDA host-length experiment on SM120

Experimental result: TTFT improved by 6.33%–12.52% in the tested setup.
Strict numerical equivalence is not established; the unchanged baseline is
also nondeterministic. See the numerical checks below before adopting it.

Normal KDA prefill previously evaluated `int(query_start_loc[-1])` in every
KDA layer. On this 45-layer model, 34 KDA layers read the same GPU scalar in
each prefill chunk. The fix computes the logical token count from
`extend_seq_lens_cpu` when constructing forward metadata and shares it across
layers. GPU-only callers retain a fallback that reads the scalar once per
metadata instance. Padding is still removed before KDA kernels and restored
as zero output afterward. KDA equations, state types, weights, and KV types
are unchanged.

## Source and configuration

- Image: `lmsysorg/sglang:v0.5.20`, source baseline
  `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
- Common prerequisite: `544005cb63d05ed9aa40e18cb9bbe30aed0bfc9d`, which selects
  `block_I=32`, `num_stages=1`, and `threads=128` for TileLang NoPE attention
  on SM120. Both variants use this prerequisite.
- KDA change: `daebd2a4f0b3704a34139f3f9002398800ece80f`.
- Hardware: 8 RTX PRO 5000 72 GB GPUs (SM120), PCIe, no NVLink; two EPYC 9K65
  sockets. No additional manual NUMA binding.
- Native FP8 model weights; BF16 KV; TP8; one running request; context limit
  1,048,576; prefill chunk and token budget 2,048; static memory fraction 0.88.
- DSA prefill and decode: TileLang. KDA, MoE, and FP8 GEMM: Triton. Decode:
  full CUDA Graph with batch size 1. Prefill CUDA Graph disabled.

Run A with the common prerequisite and B with the prerequisite plus the KDA
change. Restart between variants, retain the same launch arguments, and
verify `/server_info` before each run. Do not compare B against an older
attention backend and attribute that combined change to this fix.

## Regression tests

Run in the configured SGLang environment:

```bash
python3 test/registered/unit/layers/attention/test_kda_logical_num_tokens.py
python3 test/registered/unit/layers/attention/test_mamba_track_state_dtype.py
```

The first test covers host-mirror reuse across 34 layers, GPU-only fallback,
zero length, ragged batches, padded physical lengths, fresh metadata, and
unchanged padding behavior in `KDAAttnBackend.forward_extend`. The second
checks existing shared Mamba metadata behavior. These are targeted tests;
they do not establish correctness for every distributed or speculative mode.

## Request measurements

Use `benchmark/bench_kda_host_length.py` with the tokenizer mounted at `/model`
or supply `--model-path`. Run the following against each isolated server:

```bash
python3 benchmark/bench_kda_host_length.py \
  --model-path /model --output-dir /results/A1 --phase A1 --repeat 1 \
  --warmup --lengths 31744 130048 261120 523264 1047552

python3 benchmark/bench_kda_host_length.py \
  --model-path /model --output-dir /results/A1-quality --phase A1-quality \
  --mode quality
```

Order: A1, B1, B2, A2. Use `--repeat 2` for B2 and A2, and warm up A1, B1,
and the restarted A2 server. Performance requests use 128 output tokens,
temperature 0, `ignore_eos=true`, deterministic input IDs, and a unique cache
salt. The client asserts zero cached tokens, the exact input/output lengths,
and that the first streamed event contains one generated token. JSON
serialization occurs before timing. TTFT includes local HTTP and scheduler
overhead; decode speed is the remaining 127 tokens divided by their elapsed
streaming time. Warmups and profiler requests are excluded from comparisons.

Compare paired input and output SHA-256 values, not just generated text.
Quality mode records three repetitions of 4K/128K log probabilities and a
128K middle-document
retrieval response. Compare token IDs exactly and log probabilities with
explicit tolerances. Passing these checks supports this tested configuration;
it is not a general model accuracy benchmark.

## Profiling

Profile separate requests after timing. Use three-forward early, middle, and
late prefill windows plus a decode window on all eight ranks. Record actual
query and KV lengths, `aten::item`, four-byte DtoH copies, and
`cudaStreamSynchronize`. Count CPU annotations separately from GPU annotation
copies. The client's optional `--mode profile` requires an experiment-side
forward-progress hook and flag; ordinary performance/quality modes require
no server instrumentation.

Removing a CPU synchronization point may move its wait to a later dependency.
A profiler's accumulated CPU wait is not an estimate of saved TTFT. Report
end-to-end timing separately from synchronization counts, and keep profiled
requests out of timing statistics.

## Observed results (2026-10-17)

Performance and request completion were verified on the configuration above.
Strict numerical equivalence is **not established**: the unchanged baseline
also varies across identical requests. Keep this as an experimental branch;
the experiment restored the original service afterward.

The table uses the median of two observations per variant and input length.
All 20 timed requests completed with 128 output tokens and zero cached tokens.
TTFT reduction is `(A - B) / A`; decode speed uses the same streaming definition
for both variants.

| Input tokens | A TTFT (s) | B TTFT (s) | TTFT reduction | A decode (token/s) | B decode (token/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 31,744 | 3.406 | 3.190 | 6.33% | 76.557 | 76.459 |
| 130,048 | 14.349 | 13.430 | 6.41% | 75.951 | 75.835 |
| 261,120 | 30.362 | 28.287 | 6.83% | 75.344 | 75.257 |
| 523,264 | 67.803 | 62.098 | 8.41% | 73.943 | 73.901 |
| 1,047,552 | 166.533 | 145.686 | 12.52% | 71.193 | 71.173 |

Decode changes ranged from -0.15% to -0.03%, with no measured decode gain.
These are two observations per variant, not a confidence interval. A1 timing
completed before an ancillary retrieval-client failure; A was reloaded with
identical production sources to finish quality checks and profiling. B1/B2
then ran in one process, followed by a restarted A2. The performance request
implementation was unchanged when the quality client was repaired.

Across all eight ranks, the three-forward early, middle, and late prefill
windows each lost exactly 102 four-byte DtoH copies and 102 stream
synchronizations (34 KDA layers × 3 forwards). For example, the late window
changed from 168 to 66 `cudaStreamSynchronize` events per rank and from 123 to
21 four-byte DtoH copies. Other copies and synchronization points remain.
Decode windows showed no change. Both variants produced 32 trace files.

### Numerical checks and limits

Paired timing requests had identical input IDs. Only 1 of 10 pairs had an
identical full generated-token hash; this fails a strict token-parity gate.
The three 128K middle-document retrieval checks returned the expected code.

Teacher-forced comparisons use the same last 32 input tokens. Once generated
prefixes differ, their later output-token scores are not comparable. The
unchanged A2 process was repeated three times per length; A1 and B1 each had
one such request per length:

| Input | Baseline A1↔A2 max absolute score difference (range over 3 pairs) | Baseline A2↔A2 (3 pairs) | A↔B (4 pairs) |
| ---: | ---: | ---: | ---: |
| 4K | 0.540–1.580 | 0.667–1.039 | 0.367–1.495 |
| 128K | 0.586–1.295 | 0.320–0.717 | 0.394–0.750 |

The cross-variant differences overlap the observed baseline variation. This
small sample does not establish a new accuracy regression or rule one out.
It does not justify a claim of bitwise equivalence, unchanged perplexity, or
zero accuracy loss. Existing TileLang mHC data-race compiler warnings remain;
the experiment did not establish their relationship to numerical variation.

No timed request failed, and no fatal CUDA OOM, illegal memory access, or NCCL
error was observed. Both variants logged allocator OOM/retry warnings. The
5-second GPU samples reached only 10 MiB free on A2; the memory configuration
has little headroom. Request completion does not imply ample memory margin
or production stability under concurrency. Default launch scripts and the
default KDA source remained unchanged.
