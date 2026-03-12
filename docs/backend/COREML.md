# Apple Silicon NPU (CoreML) in llama.cpp

This fork documents the Apple Silicon NPU path in `llama.cpp` when running on macOS systems with Neural Engine hardware.
Busy requires a checkout with native CoreML backend support to use NPU on Apple Silicon.
On Apple Silicon with NPU, Busy fails fast if `GGML_COREML` is not enabled.
Metal fallback is disabled by default and can only be enabled explicitly.

## What This Fork Actually Implements

This repository does not carry a separate hidden NPU branch. The Apple Silicon work was landed directly on `main` in the fork history.

In the current tree, `GGML_COREML` enables a `COREML` backend registration path that:

- detects Apple Silicon / Neural Engine-capable hardware
- registers `COREML*` devices ahead of Metal in backend selection
- exposes a CoreML/NPU-specific execution choice to downstream tooling such as Busy

In this implementation, the `COREML` backend delegates graph planning, graph execution, buffer handling, and backend initialization to the Metal backend already present in the tree. In other words, `COREML0` in this fork is a CoreML/NPU-targeted device-selection and routing layer over Metal-backed execution, not a separate standalone non-Metal executor.

That distinction matters because the benchmarked NPU path in this repo is real and reproducible, but its implementation model is easy to misremember if you only look at launch flags or product-level docs.

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

Run the server with a model path:

```sh
build/bin/llama-server \
  -m /path/to/model.gguf \
  -c 4096 \
  --port 8082 \
  --reasoning-format deepseek \
  --media-path /tmp \
  --n-gpu-layers 20 \
  --main-gpu 0
```

## Interpreting `--device COREML0` And `--n-gpu-layers 999`

The measured M1 Max path in this fork used:

- `--device COREML0`
- `--n-gpu-layers 999`

`COREML0` selects the fork's CoreML/NPU-specific backend entry point described above.

`--n-gpu-layers 999` was used as a practical "full offload request" setting in the benchmarked runs so the runtime could offload as aggressively as the implementation and hardware allowed. In this fork's measured runs, that materially outperformed partial offload.

For the recorded benchmark baseline and exact launch command, see [M1_MAX_BENCHMARKS.md](../../M1_MAX_BENCHMARKS.md).

## Related Notes

- Busy keeps Metal fallback as an explicit opt-in path for CoreML/NPU requests:
  - set `LLAMA_COREML_FALLBACK=1` (or `--llama-coreml-fallback`) when using `setup.py`.
- By default, if NPU is available and `-DGGML_COREML=ON` is absent, setup will fail with an error.
- For a one-command path from Busy, pass `--llama-backend npu` and keep the `llama.cpp` source checkout aligned to a CoreML-capable revision.
