# Technique Matrix Follow-Up

This bundle captures the consolidated technique-matrix run on the local M3 Max
using `ggml-org_gemma-3-1b-it-Q4_K_M.gguf`.

Files:

- `technique_matrix.json`
  - full raw case data, consolidated per-technique effects, and granularity recommendation
- `technique_matrix.md`
  - human-readable summary of the same run

## M1 Max Reproduction

Run the same profiler on the M1 Max from `sam/technique-matrix` after building
`build-apple-silicon` there:

```sh
python3 scripts/profile_technique_matrix.py \
  --server-bin build-apple-silicon/bin/llama-server \
  --model "$HOME/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf" \
  --device MTL0 \
  --parallel 4 \
  --kv-unified \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --n-gpu-layers 999 \
  --n-predict 32 \
  --rounds 2 \
  --trials 2 \
  --output json \
  > artifacts/perf-comparisons/technique-matrix-m1-gemma1b/technique_matrix.json
```

Then render a readable summary:

```sh
python3 - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('artifacts/perf-comparisons/technique-matrix-m1-gemma1b/technique_matrix.json').read_text())
lines = [
    '# Technique Matrix M1 Max Gemma 1B',
    '',
    f"- model: `{p['model']}`",
    f"- device: `{p['device']}`",
    f"- parallel: `{p['parallel']}`",
    f"- kv_unified: `{p['kv_unified']}`",
    '',
    '## Granularity',
    '',
    f"- recommendation: `{p['analysis']['granularity_recommendation']['mode']}`",
    f"- reason: {p['analysis']['granularity_recommendation']['reason']}",
]
Path('artifacts/perf-comparisons/technique-matrix-m1-gemma1b/technique_matrix.md').write_text('\n'.join(lines) + '\n')
PY
```

Commit the resulting `artifacts/perf-comparisons/technique-matrix-m1-gemma1b/`
directory back to this branch so the recommendation can be compared directly
against the M3 Max run.
