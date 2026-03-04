# Apple Silicon NPU (CoreML) in llama.cpp

This fork documents the Apple Silicon NPU path in `llama.cpp` when running on macOS systems with Neural Engine hardware.
On Apple Silicon, Busy defaults to CoreML/NPU when available and falls back to Metal when unavailable unless explicitly disabled by environment.

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

- The Busy integration keeps a fallback to Metal when CoreML is unavailable.
- If you need to disable this fallback at startup, set `LLAMA_COREML_FALLBACK=0`.
- For a one-command path from Busy, pass `--llama-backend npu` and keep the `llama.cpp` source checkout aligned to this fork.
