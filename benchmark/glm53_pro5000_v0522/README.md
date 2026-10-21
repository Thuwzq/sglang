# GLM-5.3-Flash on PRO5000: v0.5.22 bring-up

This branch starts at official tag `v0.5.22`, commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`, matching the imported image build.
The v0.5.20 branch `glm5.3-flash-fp8-pro5000` remains the previous optimized
baseline. This branch currently contains only the SM120 NoPE TileLang small-tile
repair and the minimal deployment profile. It is not yet a replacement for
that full baseline.

## Kernel repair

The stock NoPE sparse attention tile requested 151,552 bytes of dynamic shared
memory during decode CUDA Graph capture and failed to launch on RTX PRO 5000.
For SM120 and `tail_dim == 0`, use `block_I=32`, `num_stages=1`, `threads=128`.
Other devices and non-NoPE inputs retain the existing tile defaults.

The repository file differs from the deployed file only in line wrapping;
their full parsed AST hashes match. Both source hashes and the common AST hash
are recorded in [validation-observed.json](validation-observed.json).
This check establishes source correspondence, not model numerical equivalence.

## Recorded profile

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
or concurrency matrix has been validated on v0.5.22. The small smoke does not
establish full numerical accuracy or long-running stability.

## Open issue and migration queue

Enabling `--enable-attn-tp-input-scattered` reached warmup but failed in
`glm5_next.py` at the FFN boundary preparation, followed by `reduce_scatter`:

```text
ValueError: input tensor must be the same size as output size times world size
```

The same profile with SP disabled completed warmup and the short protocol smoke.
SP remains disabled as a workaround; its root cause and repair are pending.
The earlier shared-memory capture error and this SP error belong to separate
failed instances and are not new errors in the successful instance.

| Previous change | v0.5.22 status |
| --- | --- |
| SM120 NoPE small tile | Migrated and short device smoke passed |
| KDA host-length reuse | Pending upstream audit and migration if needed |
| Hybrid HiCache index geometry/restore/cleanup | Pending upstream audit and migration if needed; HiCache disabled |
| Packed MTP row guard and recurrent checkpoint ownership | Pending upstream audit and migration if needed; MTP disabled |
| Mixed-extend grammar barrier | Pending upstream audit and migration if needed; MTP/mixed chunk disabled |
| Adaptive MTP Graph pruning | Pending upstream audit and migration if needed; MTP disabled |
| Native FlashInfer GLM NoPE + FP8 KV | Not implemented or validated by this branch |

Port each remaining change independently after checking the new upstream code.
Neither all old patches nor their test results are inherited by this branch.
Publishing these files does not restart the running service or resume monitoring.

Local publication checks passed: Ruff 0.15.1 formatting, the repository's selected
Python lint rules plus import ordering, Python syntax parsing, launch-helper dry
run, deployment AST/argv hash comparisons, and `git diff --check`. The helper is
a new host-path adapter verified locally in dry-run mode; the device observations
above came from the original launch command, not a fresh run of this helper.
