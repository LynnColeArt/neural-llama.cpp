# Apple Silicon Benchmark Notes

## Scope

This document now carries two related but distinct Apple Silicon performance
artifacts:

1. the original M1 Max `llama-server` baseline on `qwen3.5-0.8B-Q4_K_M.gguf`
2. the newer M3 Max backend-selection validation bundle on
   `gemma-3-1b-it-Q4_K_M.gguf`

They are both useful, but they are not apples-to-apples. The model, workload,
and context regime differ, so cross-machine comparisons should focus on backend
behavior and default-policy correctness unless the same model and prompt path
are rerun on both systems.

## M1 Max Baseline

These are local baseline performance results for the `neural-llama.cpp` fork
on Apple Silicon M1 Max using `qwen3.5-0.8B-Q4_K_M.gguf` with CoreML/ANE
execution.

- Fork commit: `0ee8108abb`
- Model path: `models/qwen3.5-0.8b.Q4_K_M.gguf`
- Model context window used: `n_ctx_train = 262144` (set with `-c 262144`)
- Runtime mode: text-only and multimodal requests against `llama-server`

## Server launch (reproducible)

```bash
vendor/llama.cpp/build/bin/llama-server \
  -m models/qwen3.5-0.8b.Q4_K_M.gguf \
  --port 8082 \
  --reasoning-format deepseek \
  -c 262144 \
  --n-gpu-layers 999 \
  --main-gpu 0 \
  --device COREML0 \
  --media-path /tmp
```

For multimodal runs, include the matching projector:

```bash
--mmproj models/mmproj-Qwen_Qwen3.5-0.8B-f16.gguf
```

## Benchmark methodology

- Client used: Busy benchmark helper (`scripts/run_npu_benchmark.sh` / `docs/internal/npu_benchmark.py`)
- Measurement fields recorded:
  - `ttft_s` (time-to-first-token)
  - `elapsed_s` (total completion wall time)
  - `predicted_tps`
  - `predicted_n` (actual completion tokens observed in that request)
- Text-only prompts were run with stable single-turn instruction queries.
- Multimodal prompt uses an inline base64 image input with an attached image prompt.

## Text-only results (CoreML/ANE)

| Test Label   | Max Tokens Requested | TTFT (s) | Elapsed (s) | Predicted TPS | Completion Tokens |
|--------------|---------------------|----------|-------------|---------------|------------------|
| text-1024    | 1024                | 0.0494   | 14.0123     | 73.3367       | 1024             |
| text-2048    | 2048                | 0.0491   | 24.8471     | 75.9334       | 1883             |

Observations:
- `n-gpu-layers 999` (full offload request) gives materially better throughput versus partial offload in earlier runs.
- The 2048-cap run returned fewer than requested tokens in one sample (`1883`) due early termination.

## Multimodal results (CoreML/ANE + mmproj)

| Test Label       | Max Tokens Requested | TTFT (s) | Elapsed (s) | Predicted TPS | Completion Tokens |
|------------------|---------------------|----------|-------------|---------------|------------------|
| multimodal-image  | 256                 | 1.0232   | 4.3614      | 76.6846       | 256              |

Notes:
- A valid mmproj is mandatory for image routing. If omitted, the server returns:
  - `image input is not supported - hint: if this is unexpected, you may need to provide the mmproj`
- The image benchmark above was executed with the file:
  - `/Users/sam/Downloads/hankythexmaspooh_icon_for_a_cute_JOYFUL_little_guy_named_Sparky_2ce856a4-b641-45e8-9bbe-7ad83825c326.png`

## Interpretation

- Current M1 Max runs with CoreML/ANE are in a practical 73–76 TPS range on this model for text completion under high context.
- TTFT is sub-100 ms for text and about 1.0 s for image-conditioned prompts in the measured sample.
- For apples-to-apples reruns, keep `-c 262144`, `--n-gpu-layers 999`, `--main-gpu 0`, and `--device COREML0` constant.

## M3 Max Backend Validation

The updated Apple Silicon validation harness was run on an Apple M3 Max and the
full evidence bundle is committed at:

- `artifacts/apple-silicon-validation/samuels-macbook-pro-2-20260313-193027/`

The quick-read verdict is in:

- `artifacts/apple-silicon-validation/samuels-macbook-pro-2-20260313-193027/summary.md`

That summary now includes the structured machine profile for the validation
host: model identifier, CPU split, GPU cores, ANE visibility/device count,
memory bytes, unified-memory flag, Metal family, and firmware/loader versions.

The underlying probe matrix is in:

- `artifacts/apple-silicon-validation/samuels-macbook-pro-2-20260313-193027/probe.md`
- `artifacts/apple-silicon-validation/samuels-macbook-pro-2-20260313-193027/probe.json`

### Validation setup

- Machine: Apple M3 Max
- Model: `gemma-3-1b-it-Q4_K_M.gguf`
- Tool: `scripts/run_apple_silicon_validation.sh`
- Prompt tokens: `512`
- Generation tokens: `128`
- Repetitions: `3` for the explicit backend probe, `1` for the `auto` check
- Flash attention: enabled

### M3 Max probe results

| Config           | Device  | NGL | Prompt TPS | Gen TPS |
|------------------|---------|-----|------------|---------|
| cpu              | none    | 0   | 935.49     | 112.19  |
| metal_partial20  | MTL0    | 20  | 2469.11    | 126.73  |
| metal_full999    | MTL0    | 999 | 6438.81    | 216.27  |
| coreml_partial20 | COREML0 | 20  | 2251.35    | 135.92  |
| coreml_full999   | COREML0 | 999 | 6357.99    | 192.92  |

`auto` on the same build/model landed at:

- prompt TPS: `6419.27`
- gen TPS: `219.99`
- delta vs best explicit gen TPS: `-1.95%`

### M3 Max interpretation

- On this M3 Max, the best explicit backend is `MTL0 --n-gpu-layers 999`.
- `MTL0` full offload beat `COREML0` full offload by about `15.29%`.
- `metal_full999` beat `metal_partial20` by about `64.02%`.
- `coreml_full999` beat `coreml_partial20` by about `58.66%`.
- `auto` now tracks the fastest explicit path closely enough to count as
  correct on this machine.
- Startup logs show that explicit `COREML0` still allocates KV and compute
  buffers on `MTL0`, so the current `COREML0` path is still Metal-routed in
  practice on this branch.

## Current Takeaway

- The older M1 Max baseline proves that `COREML0 --n-gpu-layers 999` was a
  practical high-performance path on that machine/model/workload.
- The newer M3 Max validation proves that a generic Apple Silicon
  `COREML`-first policy does not hold across hardware generations.
- The right cross-machine strategy in this fork is to validate backend choice
  with evidence, not assume one backend label is universally best.
