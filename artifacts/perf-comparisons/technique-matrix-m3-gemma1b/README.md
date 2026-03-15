# Technique Matrix Follow-Up

This bundle captures the consolidated technique-matrix run on the local M3 Max
using `ggml-org_gemma-3-1b-it-Q4_K_M.gguf`.

Files:

- `technique_matrix.json`
  - full raw case data, consolidated per-technique effects, and granularity recommendation
- `technique_matrix.md`
  - human-readable summary of the same run

## M1 Max Reproduction

Run this from `sam/technique-matrix` on the M1 Max:

```sh
./scripts/run_technique_matrix_capture.sh \
  --model "$HOME/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf" \
  --label technique-matrix-m1-gemma1b
```

That single command will build `llama-server` if needed, run the matrix, write
the artifact bundle, and print the recommendation to the console.

Then commit the resulting `artifacts/perf-comparisons/technique-matrix-m1-gemma1b/`
directory back to this branch so the recommendation can be compared directly
against the M3 Max run.
