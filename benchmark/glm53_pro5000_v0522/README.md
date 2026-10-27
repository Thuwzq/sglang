# GLM-5.3-Flash on PRO5000: v0.5.22 fixes and validation

This branch starts at official tag `v0.5.22`, commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`, matching the imported image build.
The v0.5.20 branch `glm5.3-flash-fp8-pro5000` retains the earlier implementation.
This branch, `glm5.3-flash-fp8-pro5000-v0.5.22`, contains the three maintained
runtime source overrides listed below. The original minimal BF16 launch helper
and its observations are preserved as a **2026-10-21 bring-up snapshot**; that
helper is not the current FP8 KV / HiCache / MTP deployment command.

## Patch inventory, audited 2026-10-27

| Source | Change | Validation |
| --- | --- | --- |
| `python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py` | SM120 NoPE small tile; avoids excessive dynamic shared memory during Graph capture | Original bring-up smoke and later deployed use |
| `python/sglang/srt/models/glm5_next.py` | Restore DSA physical TP padding before the attention boundary wraps the output | 4 CPU tests; earlier SP + mixed chunk + HiCache device checks |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | Commit pending grammar state before extend/mixed-extend sampling under speculative overlap | 6 CPU tests; earlier 49/49 structured SSE requests |

The read-only deployment audit compared all three complete parsed ASTs against
the running container: all matched. The grammar worker also matched byte for
byte; the kernel and model files have textual differences but identical ASTs.
The audit reran all 10 CPU tests against the actual deployed sources, with no
failures or errors. It did not restart the service or generate inference load.
See [fixes-validation-20261027.json](fixes-validation-20261027.json) for source
fingerprints, test scope and limitations.

### SP DSA padding

With attention TP input scattering, the DSA path can return only logical token
rows while the subsequent reduce-scatter and mHC residual still require the
physical padded row count. Restore zero dummy rows **before**
`attn_boundary.finish`, which may wrap the unreduced output. Reject an
unexpected loss of real rows rather than silently padding it. KDA and SP-off
paths retain their existing behavior.

The regression covers logical lengths 6, 7, 9, 4,103 and 4,096, unchanged real
rows and routing indices, zero dummy rows, SP-off, KDA and invalid row loss.
The earlier device experiment passed 9 mixed SSE requests and 32 concurrent SSE
requests, plus a host-cache restore probe. These are functional checks, not a
broad model-accuracy certification.

### Speculative mixed-extend grammar barrier

The v0.5.22 worker accepted a `grammar_barrier` callback but skipped it in the
extend/mixed-extend dispatch. With overlapping speculative work, a request could
sample from grammar state that had not yet consumed its preceding accepted
tokens. Later `grammar.accept_token()` could reject the result, producing an
HTTP 200 SSE stream whose finish reason was `abort`.

Call the existing barrier before this branch samples when the batch has grammar
and a callback is available. Decode dispatch, speculative acceptance rules and
mixed-chunk enablement are unchanged. The six CPU tests exercise pending-result
ordering, both extend dispatch conditions, a new prefill grammar, no grammar,
and absent callbacks. Earlier patched GPU regression passed 49/49 structured
streams across single, concurrency-8 short/mixed-32K/mixed-128K, concurrency-16
JSON schema and concurrency-8 tool-call cases. Every stream ended legally and
included `[DONE]`; the regression log window contained no matched grammar,
abort, OOM, GPU, traceback, NCCL, watchdog or HTTP 5xx errors.

Run the CPU regressions from the repository root (PyTorch is required for the
padding test; neither test imports model GPU kernels):

```bash
python3 test/registered/unit/models/test_glm5_next_sp_padding.py -v
python3 test/registered/unit/speculative/test_eagle_mixed_extend_grammar_barrier.py -v
```

For offline stock-image deployment, use this checkout's three source overrides
at the matching paths beneath `/sgl-workspace/sglang/` as read-only bind mounts,
or build an image from this branch. The historical helper below mounts only the
kernel; it does not deploy the later model and grammar fixes.

## Kernel repair

The stock NoPE sparse attention tile requested 151,552 bytes of dynamic shared
memory during decode CUDA Graph capture and failed to launch on RTX PRO 5000.
For SM120 and `tail_dim == 0`, use `block_I=32`, `num_stages=1`, `threads=128`.
Other devices and non-NoPE inputs retain the existing tile defaults.

The repository file differs from the deployed file only in line wrapping;
their full parsed AST hashes match. Both source hashes and the common AST hash
are recorded in [validation-observed.json](validation-observed.json).
This check establishes source correspondence, not model numerical equivalence.

## Historical minimal profile, 2026-10-21

| Setting | Value |
| --- | --- |
| Hardware | 8 x RTX PRO 5000 72GB, SM120 |
| Image | `lmsysorg/sglang:v0.5.22`, imported offline |
| Weight quantization / DSA KV | FP8 / BF16 |
| TP / PP / DP / EP | 8 / 1 / 1 / 1 |
| DSA prefill / decode | TileLang / TileLang |
| KDA prefill / decode / verify | Triton / Triton / Triton |
| MoE runner / FP8 GEMM | Triton / Triton |
| Decode Graph buckets | 1, 2, 4, 8, 16, 32 |
| Prefill Graph | Disabled |
| Chunked prefill / prefill token budget | 1024 / 1024 |
| Max running requests / recurrent slots | 32 / 192 |
| Configured context / device token pool | 1,048,576 / 1,064,960 |
| Static memory fraction | 0.84 |
| SP / MTP / mixed chunk / HiCache | Disabled |

The exact server argv is in [server-args.json](server-args.json). Its canonical
JSON SHA-256 matches the device launch argv. The helper adapts host paths and
mounts this checkout's patched kernel into the stock image. It also adds
`--pull never` for offline use. It does not build or publish a new image.

To inspect the expanded command without accessing Docker or creating files:

```bash
python3 benchmark/glm53_pro5000_v0522/launch.py --dry-run
```

To reproduce on a device with the recorded image already imported, the model
present, port 8000 available, and all eight GPUs available:

```bash
python3 benchmark/glm53_pro5000_v0522/launch.py \
  --model-dir /data1/models/GLM-5.3-Flash-fp8 \
  --runtime-dir "$HOME/glm5_3flash_runtime"
```

Choose the existing runtime directory explicitly if it differs from the example.
The helper refuses an existing container name or an image ID that differs from
the recorded deployment. It never stops another container. Scripts/configuration
are under `serving/v0522-tp8-ep1-bf16-minimal/`, compilation caches under
`cache-v0522/`, and the service log under
`logs/v0522-tp8-ep1-bf16-minimal-service.log` in the selected runtime directory.
The checkout must remain available because the kernel source is a read-only bind
mount. For a previously created container, inspect it before using `docker start`.

After the log reports that the server is ready:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"What is 2 plus 2? Answer briefly."}],"max_tokens":64,"temperature":1.0,"top_p":0.95,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

`enable_thinking=false` was supplied in the smoke request, but the observed
non-stream response still included reasoning content. This record does not
establish a working thinking-mode switch. To repeat the non-stream protocol
check, set `stream` to `false` in the same payload.

## Device observations, 2026-10-21

These are short bring-up observations transcribed from the WeTERM console;
the JSON is a summary, not a downloaded raw log or a benchmark report.
Internal addresses and user-specific host paths are omitted from this directory.

| Probe | Observed result |
| --- | --- |
| Health | HTTP 200 |
| Non-stream chat | HTTP 200, correct answer 4, `finish_reason=stop`, 50 output tokens, 0.712 s total |
| SSE chat | HTTP 200, 49 chunks, valid output, `stop`, `[DONE]`, no stream errors, 0.670 s total |
| Successful instance startup log | No matched traceback, CUDA OOM, illegal access or grammar-accept failure |
| Driver-visible GPU memory after smoke | 56,579 MiB used and 16,252 MiB free on each card |

The durations are total short-request times, not TTFT or steady-state decode
throughput. Graph capture through batch 32 completed; actual concurrent batch-32
inference was not validated in this bring-up. 1M is configured, but no 1M request
or concurrency matrix was validated in this minimal bring-up. The small smoke does not
establish full numerical accuracy or long-running stability.

## Deployment scope and remaining limits

The initial SP warmup failure (`input tensor must be the same size as output
size times world size`) was subsequently addressed by the padding repair above.
The initial kernel shared-memory failure and SP failure were separate failed
instances, not errors in the successful minimal instance.

The 2026-10-27 audited running profile used TP8/PP1/EP1, FP8 DSA KV, mixed chunk,
overlap, adaptive MTP 0/1/3 and HiCache ratio 13; SP was disabled. Active and Graph
limits were 64, recurrent slots 320, requested GPU KV capacity 1,200,000 tokens
(actual page-aligned capacity 1,199,872), chunk/prefill budget 1,024 and context
limit 1,048,576. These settings describe that instance and do not promote it over
other saved baselines or override their independent capacity/performance tests.

Earlier v0.5.20 KDA, HiCache and Graph patches are not blindly copied into this
branch. Only the three v0.5.22 source changes in the inventory are maintained
here; upstream-provided behavior and unrelated scheduler experiments are outside
this patch set. FlashInfer-versus-TileLang measurements and adaptive chunk
experiments are not part of these correctness regressions.

The grammar GPU results are from the preceding deployment acceptance test, not a
new pre/post GPU A/B during this source sync. They cover inputs up to 132,043
tokens, not a 1M Agent endurance run. The CPU tests model dispatch and boundary
contracts and cannot certify GPU numerical accuracy. No throughput gain, full
model accuracy or long-duration success-rate guarantee is inferred from them.
Publishing this branch does not restart the running service or resume monitoring.
