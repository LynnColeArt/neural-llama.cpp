# M1 Max NPU Benchmarks

## Scope

These are local baseline performance results for the `neural-llama.cpp` fork on Apple Silicon M1 Max using
`qwen3.5-0.8B-Q4_K_M.gguf` with CoreML/ANE execution.

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
