# Apple Silicon NPU (CoreML) in llama.cpp

This fork documents the Apple Silicon NPU path in `llama.cpp` when running on macOS systems with Neural Engine hardware.
Busy requires a checkout with native CoreML backend support to use NPU on Apple Silicon.
On Apple Silicon with NPU, Busy fails fast if `GGML_COREML` is not enabled.
Metal fallback is disabled by default and can only be enabled explicitly.

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

## Related Notes

- Busy keeps Metal fallback as an explicit opt-in path for CoreML/NPU requests:
  - set `LLAMA_COREML_FALLBACK=1` (or `--llama-coreml-fallback`) when using `setup.py`.
- By default, if NPU is available and `-DGGML_COREML=ON` is absent, setup will fail with an error.
- For a one-command path from Busy, pass `--llama-backend npu` and keep the `llama.cpp` source checkout aligned to a CoreML-capable revision.
