# Apple Silicon NPU (CoreML) in llama.cpp

This fork documents the Apple Silicon NPU path in `llama.cpp` when running on macOS systems with Neural Engine hardware.
Busy requires a checkout with native CoreML backend support to use NPU on Apple Silicon.
On Apple Silicon with NPU, Busy fails fast if `GGML_COREML` is not enabled.

## Important runtime note

This tree exposes `COREML` and `METAL` as separate device names, but the current
`COREML` backend is still routed through the Metal-backed execution path in this
fork. Do not assume `COREML0` is the fastest option on every Apple Silicon
generation.

The generic default in this fork now prefers `METAL`. If you want the best
backend choice for a specific machine and model, run:

```sh
python3 scripts/apple_silicon_backend_probe.py \
  --model /path/to/model.gguf
```

On the local M3 Max validation run that motivated this change, `MTL0` with full
offload outperformed `COREML0` with full offload on Gemma 3 1B Q4_K.

## Requirements

- Apple Silicon Mac (M-series or later, running macOS with CoreML-capable runtime)
- Xcode Command Line Tools and a supported CMake/Ninja toolchain
- The upstream `llama.cpp`/fork source and generated Metal assets from a recent build

## Build

Use the standard Metal/CoreML build flags:

```sh
cmake -S . -B build \
  -DGGML_METAL=ON \
  -DGGML_COREML=ON \
  -DLLAMA_CURL=OFF

cmake --build build --config Release -j4
```

If you only want a Metal-only build (no CoreML backend), omit `-DGGML_COREML=ON`.

## Run

Start with an explicit device while validating a machine:

```sh
build/bin/llama-server \
  -m /path/to/model.gguf \
  -c 4096 \
  --port 8082 \
  --reasoning-format deepseek \
  --media-path /tmp \
  --device MTL0 \
  --n-gpu-layers 999
```

Then compare against CoreML explicitly:

```sh
build/bin/llama-server \
  -m /path/to/model.gguf \
  -c 4096 \
  --port 8082 \
  --reasoning-format deepseek \
  --media-path /tmp \
  --device COREML0 \
  --n-gpu-layers 999
```

## Related Notes

- Busy keeps Metal fallback as an explicit opt-in path for CoreML/NPU requests:
  - set `LLAMA_COREML_FALLBACK=1` (or `--llama-coreml-fallback`) when using `setup.py`.
- By default, if NPU is available and `-DGGML_COREML=ON` is absent, setup will fail with an error.
- For a one-command path from Busy, pass `--llama-backend npu` and keep the `llama.cpp` source checkout aligned to a CoreML-capable revision.
- `--n-gpu-layers 999` remains the right first full-offload probe on Apple Silicon because it lets the runtime clamp to the hardware/model limits instead of baking in an M1-era layer count.
