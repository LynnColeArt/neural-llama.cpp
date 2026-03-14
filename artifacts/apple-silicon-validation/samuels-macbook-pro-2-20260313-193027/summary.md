# Apple Silicon Validation Summary

## System
- chip: Apple M3 Max
- cpu_cores: 16 (12 Performance and 4 Efficiency)
- gpu_cores: 40
- memory: 128 GB
- metal_support: Metal 4
- model: /Users/sam/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf

## Device Exposure
- MTL0: Apple M3 Max (110100 MiB, 110100 MiB free)
- COREML0: Apple Silicon NPU (CoreML backend) (110100 MiB, 110100 MiB free)
- BLAS: Accelerate (0 MiB, 0 MiB free)

## Best Explicit Backend
- config: metal_full999
- device: MTL0
- n_gpu_layers: 999
- prompt_tps: 6454.74
- gen_tps: 224.36

## Auto Path
- devices: auto
- backends: MTL,COREML,BLAS
- prompt_tps: 6419.27
- gen_tps: 219.99
- delta_vs_best_gen_tps: -4.37 (-1.95%)

## Startup Log Readout
- auto using devices: MTL0, COREML0
- auto KV devices: MTL0
- auto compute devices: CPU, MTL0
- explicit MTL0 KV devices: MTL0
- explicit COREML0 KV devices: MTL0
- explicit COREML0 compute devices: CPU, MTL0
- explicit COREML0 routes through MTL0 buffers: yes

## Interpretation
- auto_matches_best_explicit: yes
- coreml_full999_faster_than_metal_full999: no
- inspect `probe.md`, `probe.json`, and the startup logs for the full evidence trail