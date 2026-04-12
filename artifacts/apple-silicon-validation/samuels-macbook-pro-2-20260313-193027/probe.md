# Apple Silicon Backend Probe

- platform: macOS-26.3.1-arm64-arm-64bit
- machine: arm64
- model_name: MacBook Pro
- model_identifier: Mac15,9
- model_number: Z1AF001AJLL/A
- chip: Apple M3 Max
- cpu_cores: 16 (12 Performance and 4 Efficiency)
- cpu_total_cores: 16
- cpu_performance_cores: 12
- cpu_efficiency_cores: 4
- cpu_physical_cores: 16
- cpu_logical_cores: 16
- gpu_name: Apple M3 Max
- gpu_cores: 40
- npu_visible: True
- npu_device_count: 1
- npu_visible_devices: ane0
- npu_core_count: not_exposed_by_public_macos_apis
- memory: 128 GB
- memory_bytes: 137438953472
- memory_bytes_usable: 135455965184
- unified_memory: True
- metal_support: Metal 4
- system_firmware: 13822.81.10
- os_loader_version: 13822.81.10
- model: /Users/sam/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf
- prompt_tokens: 512
- gen_tokens: 128
- repetitions: 3
- flash_attn: True

| Config | Device | NGL | Prompt TPS | Gen TPS |
|--------|--------|-----|------------|---------|
| cpu | none | 0 | 829.27 | 116.88 |
| metal_partial20 | MTL0 | 20 | 2469.70 | 158.14 |
| metal_full999 | MTL0 | 999 | 6371.23 | 216.89 |
| coreml_partial20 | COREML0 | 20 | 2460.91 | 115.27 |
| coreml_full999 | COREML0 | 999 | 6333.37 | 197.56 |

Recommended default: `MTL0` with `--n-gpu-layers 999` (gen 216.89 tok/s, prompt 6371.23 tok/s).
