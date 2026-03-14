# Apple Silicon Backend Probe

- platform: macOS-26.3.1-arm64-arm-64bit
- machine: arm64
- chip: Apple M3 Max
- cpu_cores: 16 (12 Performance and 4 Efficiency)
- memory: 128 GB
- gpu_cores: 40
- metal_support: Metal 4
- model: /Users/sam/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf
- prompt_tokens: 512
- gen_tokens: 128
- repetitions: 3
- flash_attn: True

| Config | Device | NGL | Prompt TPS | Gen TPS |
|--------|--------|-----|------------|---------|
| cpu | none | 0 | 935.49 | 112.19 |
| metal_partial20 | MTL0 | 20 | 2469.11 | 126.73 |
| metal_full999 | MTL0 | 999 | 6438.81 | 216.27 |
| coreml_partial20 | COREML0 | 20 | 2251.35 | 135.92 |
| coreml_full999 | COREML0 | 999 | 6357.99 | 192.92 |

Recommended default: `MTL0` with `--n-gpu-layers 999` (gen 216.27 tok/s, prompt 6438.81 tok/s).
