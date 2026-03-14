#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build-apple-silicon"
SERVER_BIN="$BUILD_DIR/bin/llama-server"
BENCH_BIN="$BUILD_DIR/bin/llama-bench"
PORT_BASE=8098
MODEL_PATH=""
ARTIFACT_DIR=""
PROBE_REPETITIONS=3
AUTO_BENCH_REPETITIONS=1

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_apple_silicon_validation.sh --model /path/to/model.gguf [--artifacts-dir /path/to/output]

This script captures:
  - system profile
  - git revision
  - server device list
  - backend probe results (json + markdown)
  - default auto benchmark
  - startup logs for auto, MTL0, and COREML0
  - an evidence summary that compares auto vs explicit backends
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACT_DIR="${2:-}"
      shift 2
      ;;
    --probe-repetitions)
      PROBE_REPETITIONS="${2:-}"
      shift 2
      ;;
    --auto-bench-repetitions)
      AUTO_BENCH_REPETITIONS="${2:-}"
      shift 2
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

if [[ ! -x "$SERVER_BIN" || ! -x "$BENCH_BIN" ]]; then
  echo "Expected built binaries under $BUILD_DIR; build llama-server and llama-bench first." >&2
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This validation script is intended for Apple Silicon macOS hosts." >&2
  exit 1
fi

HOSTNAME_SAFE="$(hostname -s | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9._-')"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

if [[ -z "$ARTIFACT_DIR" ]]; then
  ARTIFACT_DIR="$ROOT_DIR/artifacts/apple-silicon-validation/${HOSTNAME_SAFE}-${TIMESTAMP}"
fi

mkdir -p "$ARTIFACT_DIR"

capture_startup_log() {
  local label="$1"
  local port="$2"
  shift 2
  local log_file="$ARTIFACT_DIR/startup_${label}.log"

  (
    cd "$ROOT_DIR"
    exec "$SERVER_BIN" \
      -m "$MODEL_PATH" \
      --host 127.0.0.1 \
      --port "$port" \
      --ctx-size 8192 \
      --reasoning-format none \
      --no-webui \
      --no-jinja \
      "$@" \
      >"$log_file" 2>&1
  ) &
  local server_pid=$!

  python3 - "$log_file" <<'PY'
import pathlib
import sys
import time

log_path = pathlib.Path(sys.argv[1])
deadline = time.time() + 120
needles = (
    "main: server is listening on http://127.0.0.1:",
    "main: model loaded",
)

while time.time() < deadline:
    if log_path.exists():
        text = log_path.read_text(errors="ignore")
        if all(needle in text for needle in needles):
            sys.exit(0)
    time.sleep(0.5)

sys.exit(1)
PY

  kill -INT "$server_pid" >/dev/null 2>&1 || true
  wait "$server_pid" >/dev/null 2>&1 || true
}

{
  echo "model=$MODEL_PATH"
  echo "build_dir=$BUILD_DIR"
  echo "timestamp=$TIMESTAMP"
  echo "probe_repetitions=$PROBE_REPETITIONS"
  echo "auto_bench_repetitions=$AUTO_BENCH_REPETITIONS"
} > "$ARTIFACT_DIR/run_meta.txt"

python3 - "$ARTIFACT_DIR/system_profile.txt" <<'PY'
import pathlib
import subprocess
import sys

output_path = pathlib.Path(sys.argv[1])
text = subprocess.check_output(
    ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
    text=True,
)
redacted_lines = []
for line in text.splitlines():
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if stripped.startswith("Serial Number (system):"):
        redacted_lines.append(f"{indent}Serial Number (system): [redacted]")
    elif stripped.startswith("Hardware UUID:"):
        redacted_lines.append(f"{indent}Hardware UUID: [redacted]")
    elif stripped.startswith("Provisioning UDID:"):
        redacted_lines.append(f"{indent}Provisioning UDID: [redacted]")
    else:
        redacted_lines.append(line)
output_path.write_text("\n".join(redacted_lines) + "\n")
PY
sw_vers > "$ARTIFACT_DIR/sw_vers.txt"
git -C "$ROOT_DIR" rev-parse HEAD > "$ARTIFACT_DIR/git_rev.txt"

"$SERVER_BIN" --list-devices > "$ARTIFACT_DIR/server_list_devices.txt" 2>&1

python3 "$ROOT_DIR/scripts/apple_silicon_backend_probe.py" \
  --model "$MODEL_PATH" \
  --repetitions "$PROBE_REPETITIONS" \
  --output json \
  > "$ARTIFACT_DIR/probe.json"

python3 "$ROOT_DIR/scripts/apple_silicon_backend_probe.py" \
  --model "$MODEL_PATH" \
  --repetitions "$PROBE_REPETITIONS" \
  --output md \
  > "$ARTIFACT_DIR/probe.md"

"$BENCH_BIN" \
  -m "$MODEL_PATH" \
  -p 512 \
  -n 128 \
  -r "$AUTO_BENCH_REPETITIONS" \
  -o json \
  -fa 1 \
  -ngl 999 \
  > "$ARTIFACT_DIR/bench_auto.json" 2>&1

capture_startup_log auto "$PORT_BASE" --n-gpu-layers 999
capture_startup_log mtl0 "$((PORT_BASE + 1))" --device MTL0 --n-gpu-layers 999
capture_startup_log coreml0 "$((PORT_BASE + 2))" --device COREML0 --n-gpu-layers 999

python3 - "$ARTIFACT_DIR" <<'PY'
import json
import pathlib
import re
import sys

artifact_dir = pathlib.Path(sys.argv[1])
probe = json.loads((artifact_dir / "probe.json").read_text())
bench_text = (artifact_dir / "bench_auto.json").read_text()
bench_json_start = bench_text.find("[")
bench_rows = json.loads(bench_text[bench_json_start:])
auto_gen = next(row for row in bench_rows if row["n_gen"] > 0)
auto_prompt = next(row for row in bench_rows if row["n_prompt"] > 0)
best = probe["recommended"]
results_by_name = {row["name"]: row for row in probe["results"]}

devices = []
for line in (artifact_dir / "server_list_devices.txt").read_text().splitlines():
    stripped = line.strip()
    if not stripped:
        continue
    if stripped.startswith(("MTL", "COREML", "BLAS")):
        devices.append(stripped)

def log_facts(name: str) -> dict[str, object]:
    text = (artifact_dir / f"startup_{name}.log").read_text(errors="ignore")
    using_devices = [match.strip() for match in re.findall(r"using device ([^\(\n]+)", text)]
    kv_devices = sorted(set(re.findall(r"llama_kv_cache:\s+([A-Z0-9]+) KV buffer size", text)))
    compute_devices = sorted(set(re.findall(r"sched_reserve:\s+([A-Z0-9]+) compute buffer size", text)))
    return {
        "using_devices": using_devices,
        "kv_devices": kv_devices,
        "compute_devices": compute_devices,
        "routes_coreml_through_mtl": "MTL0" in kv_devices or "MTL0" in compute_devices,
    }

auto_facts = log_facts("auto")
mtl_facts = log_facts("mtl0")
coreml_facts = log_facts("coreml0")

delta = auto_gen["avg_ts"] - best["gen_tps"]
delta_pct = (delta / best["gen_tps"] * 100.0) if best["gen_tps"] else 0.0
auto_matches_best = auto_gen["avg_ts"] >= (best["gen_tps"] * 0.97)
summary = artifact_dir / "summary.md"
system_fields = [
    ("model_name", "model_name"),
    ("model_identifier", "model_identifier"),
    ("model_number", "model_number"),
    ("chip", "chip"),
    ("cpu_cores", "cpu_cores"),
    ("cpu_total_cores", "cpu_total_cores"),
    ("cpu_performance_cores", "cpu_performance_cores"),
    ("cpu_efficiency_cores", "cpu_efficiency_cores"),
    ("cpu_physical_cores", "cpu_physical_cores"),
    ("cpu_logical_cores", "cpu_logical_cores"),
    ("gpu_name", "gpu_name"),
    ("gpu_cores", "gpu_cores"),
    ("npu_visible", "npu_visible"),
    ("npu_device_count", "npu_device_count"),
    ("npu_visible_devices", "npu_visible_devices"),
    ("npu_core_count", "npu_core_count"),
    ("memory", "memory"),
    ("memory_bytes", "memory_bytes"),
    ("memory_bytes_usable", "memory_bytes_usable"),
    ("unified_memory", "unified_memory"),
    ("metal_support", "metal_support"),
    ("system_firmware", "system_firmware"),
    ("os_loader_version", "os_loader_version"),
]
system_lines = []
for key, label in system_fields:
    value = probe["system"].get(key)
    if value is None:
        continue
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    system_lines.append(f"- {label}: {value}")

summary.write_text(
    "\n".join(
        [
            "# Apple Silicon Validation Summary",
            "",
            "## System",
            *system_lines,
            f"- model: {probe['model']}",
            "",
            "## Device Exposure",
            *[f"- {line}" for line in devices],
            "",
            "## Best Explicit Backend",
            f"- config: {best['name']}",
            f"- device: {best['device']}",
            f"- n_gpu_layers: {best['n_gpu_layers']}",
            f"- prompt_tps: {best['prompt_tps']:.2f}",
            f"- gen_tps: {best['gen_tps']:.2f}",
            "",
            "## Auto Path",
            f"- devices: {auto_gen['devices']}",
            f"- backends: {auto_gen['backends']}",
            f"- prompt_tps: {auto_prompt['avg_ts']:.2f}",
            f"- gen_tps: {auto_gen['avg_ts']:.2f}",
            f"- delta_vs_best_gen_tps: {delta:.2f} ({delta_pct:.2f}%)",
            "",
            "## Startup Log Readout",
            f"- auto using devices: {', '.join(auto_facts['using_devices']) or 'none'}",
            f"- auto KV devices: {', '.join(auto_facts['kv_devices']) or 'none'}",
            f"- auto compute devices: {', '.join(auto_facts['compute_devices']) or 'none'}",
            f"- explicit MTL0 KV devices: {', '.join(mtl_facts['kv_devices']) or 'none'}",
            f"- explicit COREML0 KV devices: {', '.join(coreml_facts['kv_devices']) or 'none'}",
            f"- explicit COREML0 compute devices: {', '.join(coreml_facts['compute_devices']) or 'none'}",
            f"- explicit COREML0 routes through MTL0 buffers: {'yes' if coreml_facts['routes_coreml_through_mtl'] else 'no'}",
            "",
            "## Interpretation",
            f"- auto_matches_best_explicit: {'yes' if auto_matches_best else 'no'}",
            f"- coreml_full999_faster_than_metal_full999: {'yes' if results_by_name['coreml_full999']['gen_tps'] > results_by_name['metal_full999']['gen_tps'] else 'no'}",
            "- inspect `probe.md`, `probe.json`, and the startup logs for the full evidence trail",
        ]
    )
)
PY

echo "Validation artifacts written to: $ARTIFACT_DIR"
