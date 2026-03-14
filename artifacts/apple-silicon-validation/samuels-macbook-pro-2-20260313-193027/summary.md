# Apple Silicon Validation Summary

## System
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

## Device Exposure
- MTL0: Apple M3 Max (110100 MiB, 110100 MiB free)
- COREML0: Apple Silicon NPU (CoreML backend) (110100 MiB, 110100 MiB free)
- BLAS: Accelerate (0 MiB, 0 MiB free)

## Best Explicit Backend
- config: metal_full999
- device: MTL0
- n_gpu_layers: 999
- prompt_tps: 6436.61
- gen_tps: 215.36

## Auto Path
- devices: auto
- backends: MTL,COREML,BLAS
- prompt_tps: 6411.01
- gen_tps: 216.44
- delta_vs_best_gen_tps: 1.08 (0.50%)

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