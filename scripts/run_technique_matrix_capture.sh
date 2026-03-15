#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build-apple-silicon"
SERVER_BIN="$BUILD_DIR/bin/llama-server"
MODEL_PATH=""
LABEL=""
ARTIFACT_DIR=""
DEVICE="MTL0"
PARALLEL=4
N_PREDICT=32
ROUNDS=2
TRIALS=2
N_GPU_LAYERS=999
CACHE_TYPE_K="f16"
CACHE_TYPE_V="f16"
KV_UNIFIED=1
SKIP_BUILD=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_technique_matrix_capture.sh --model /path/to/model.gguf [--label m1-gemma1b]

This script will:
  1. configure/build `build-apple-silicon` if needed
  2. run the consolidated technique matrix
  3. write json + markdown artifacts
  4. print the granularity recommendation and strongest effects to the console
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --label)
      LABEL="${2:-}"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACT_DIR="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --parallel)
      PARALLEL="${2:-}"
      shift 2
      ;;
    --n-predict)
      N_PREDICT="${2:-}"
      shift 2
      ;;
    --rounds)
      ROUNDS="${2:-}"
      shift 2
      ;;
    --trials)
      TRIALS="${2:-}"
      shift 2
      ;;
    --n-gpu-layers)
      N_GPU_LAYERS="${2:-}"
      shift 2
      ;;
    --cache-type-k)
      CACHE_TYPE_K="${2:-}"
      shift 2
      ;;
    --cache-type-v)
      CACHE_TYPE_V="${2:-}"
      shift 2
      ;;
    --no-kv-unified)
      KV_UNIFIED=0
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "--model is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script is intended for Apple Silicon macOS hosts." >&2
  exit 1
fi

if [[ -z "$LABEL" ]]; then
  HOSTNAME_SAFE="$(hostname -s | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9._-')"
  LABEL="${HOSTNAME_SAFE}-$(date +%Y%m%d-%H%M%S)"
fi

if [[ -z "$ARTIFACT_DIR" ]]; then
  ARTIFACT_DIR="$ROOT_DIR/artifacts/perf-comparisons/$LABEL"
fi

mkdir -p "$ARTIFACT_DIR"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "==> Configuring build-apple-silicon"
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DGGML_METAL=ON \
    -DGGML_COREML=ON

  echo "==> Building llama-server"
  cmake --build "$BUILD_DIR" --target llama-server -j8
fi

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "llama-server not found at $SERVER_BIN" >&2
  exit 1
fi

echo "==> Running technique matrix"
CMD=(
  python3 "$ROOT_DIR/scripts/profile_technique_matrix.py"
  --server-bin "$SERVER_BIN"
  --model "$MODEL_PATH"
  --device "$DEVICE"
  --parallel "$PARALLEL"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --n-gpu-layers "$N_GPU_LAYERS"
  --n-predict "$N_PREDICT"
  --rounds "$ROUNDS"
  --trials "$TRIALS"
  --output json
)
if [[ "$KV_UNIFIED" -eq 1 ]]; then
  CMD+=(--kv-unified)
fi
"${CMD[@]}" > "$ARTIFACT_DIR/technique_matrix.json"

echo "==> Rendering summary"
python3 - "$ARTIFACT_DIR/technique_matrix.json" "$ARTIFACT_DIR/technique_matrix.md" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
p = json.loads(json_path.read_text())

lines = [
    f"# Technique Matrix {Path(json_path).parent.name}",
    "",
    f"- model: `{p['model']}`",
    f"- device: `{p['device']}`",
    f"- parallel: `{p['parallel']}`",
    f"- kv_unified: `{p['kv_unified']}`",
    f"- trials: `{p['trials']}`",
    f"- rounds: `{p['rounds']}`",
    "",
    "## Ranking",
    "",
    "| case | round wall ms | latency ms | prompt ms | gen tok/s | restore attempts | cache hit | cache admit |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for item in p["ranking"]:
    case = p["cases"][item["label"]]
    s = case["summary"]
    lines.append(
        f"| `{item['label']}` | "
        f"{s['median_followup_round_wall_ms']:.2f} | "
        f"{s['median_followup_latency_ms']:.2f} | "
        f"{s['median_followup_prompt_ms']:.2f} | "
        f"{s['median_followup_predicted_per_second']:.2f} | "
        f"{s.get('scheduler_restore_attempts_total', [0.0])[-1]:.0f} | "
        f"{s.get('prompt_cache_hit_ratio', [0.0])[-1]:.3f} | "
        f"{s.get('prompt_cache_admission_ratio', [0.0])[-1]:.3f} |"
    )

lines += [
    "",
    "## Technique Summary",
    "",
    "| technique | on median round ms | off median round ms | pairwise delta % | wins enabled | losses enabled |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]

for name, s in p["analysis"]["techniques"].items():
    lines.append(
        f"| `{name}` | "
        f"{s['on_median_followup_round_wall_ms']:.2f} | "
        f"{s['off_median_followup_round_wall_ms']:.2f} | "
        f"{s['pairwise_median_delta_round_wall_pct']:.2f}% | "
        f"{s['wins_when_enabled']} | "
        f"{s['losses_when_enabled']} |"
    )

rec = p["analysis"]["granularity_recommendation"]
lines += [
    "",
    "## Granularity",
    "",
    f"- best case: `{p['analysis']['best_case']['label']}`",
    f"- worst case: `{p['analysis']['worst_case']['label']}`",
    f"- spread: `{p['analysis']['spread_round_wall_pct']:.2f}%`",
    f"- recommendation: `{rec['mode']}`",
    f"- reason: {rec['reason']}",
]
if rec.get("keep_configurable"):
    lines.append(f"- keep configurable: `{', '.join(rec['keep_configurable'])}`")

md_path.write_text("\n".join(lines) + "\n")
PY

echo
echo "==> Recommendation"
python3 - "$ARTIFACT_DIR/technique_matrix.json" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text())
rec = p["analysis"]["granularity_recommendation"]
print(f"mode: {rec['mode']}")
print(f"reason: {rec['reason']}")
if rec.get("keep_configurable"):
    print("keep configurable:", ", ".join(rec["keep_configurable"]))
print(f"best case: {p['analysis']['best_case']['label']}")
print(f"worst case: {p['analysis']['worst_case']['label']}")
print(f"spread: {p['analysis']['spread_round_wall_pct']:.2f}%")
print("")
print("strongest measured effects:")
rows = []
for name, summary in p["analysis"]["techniques"].items():
    rows.append((abs(summary["pairwise_median_delta_round_wall_pct"]), name, summary))
for _abs_delta, name, summary in sorted(rows, reverse=True)[:4]:
    print(
        f"  {name}: "
        f"pairwise_delta={summary['pairwise_median_delta_round_wall_pct']:.2f}% "
        f"wins={summary['wins_when_enabled']} "
        f"losses={summary['losses_when_enabled']}"
    )
PY

echo
echo "Artifacts written to:"
echo "  $ARTIFACT_DIR/technique_matrix.json"
echo "  $ARTIFACT_DIR/technique_matrix.md"
