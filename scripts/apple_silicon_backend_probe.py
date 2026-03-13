#!/usr/bin/env python3

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = (
    ("cpu", ("none", "0")),
    ("metal_partial20", ("MTL0", "20")),
    ("metal_full999", ("MTL0", "999")),
    ("coreml_partial20", ("COREML0", "20")),
    ("coreml_full999", ("COREML0", "999")),
)


def read_system_profile() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    try:
        output = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
            text=True,
        )
    except Exception:
        return info

    in_hardware = False
    in_graphics = False

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Hardware Overview:":
            in_hardware = True
            in_graphics = False
            continue
        if stripped == "Graphics/Displays:":
            in_hardware = False
            in_graphics = True
            continue

        if stripped.startswith("Chip:"):
            info["chip"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Memory:"):
            info["memory"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Metal Support:"):
            info["metal_support"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Total Number of Cores:"):
            value = stripped.split(":", 1)[1].strip()
            if in_hardware:
                info["cpu_cores"] = value
            elif in_graphics:
                info["gpu_cores"] = value

    return info


def run_bench(bench_path: Path, model_path: Path, prompt_tokens: int, gen_tokens: int, repetitions: int, flash_attn: bool, config_name: str, device: str, n_gpu_layers: str) -> dict:
    cmd = [
        str(bench_path),
        "-m", str(model_path),
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-r", str(repetitions),
        "-o", "json",
        "-dev", device,
        "-ngl", n_gpu_layers,
    ]
    if flash_attn:
        cmd.extend(["-fa", "1"])

    output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    json_start = output.find("[")
    if json_start < 0:
        raise RuntimeError(f"llama-bench output for {config_name} did not contain JSON")

    data = json.loads(output[json_start:])
    prompt_row = next(row for row in data if row["n_prompt"] > 0)
    gen_row = next(row for row in data if row["n_gen"] > 0)

    return {
        "name": config_name,
        "device": device,
        "n_gpu_layers": int(n_gpu_layers),
        "prompt_tps": prompt_row["avg_ts"],
        "gen_tps": gen_row["avg_ts"],
        "prompt_samples": prompt_row["samples_ts"],
        "gen_samples": gen_row["samples_ts"],
        "gpu_info": gen_row["gpu_info"],
        "backends": gen_row["backends"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe Metal/CoreML backend performance on Apple Silicon and recommend a default policy."
    )
    parser.add_argument("--bench-path", default="build-apple-silicon/bin/llama-bench", help="Path to llama-bench")
    parser.add_argument("--model", required=True, help="Path to the GGUF model to benchmark")
    parser.add_argument("--prompt-tokens", type=int, default=512, help="Prompt tokens for llama-bench")
    parser.add_argument("--gen-tokens", type=int, default=128, help="Generated tokens for llama-bench")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per config")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention for the probe")
    parser.add_argument("--output", choices=("json", "md"), default="md", help="Output format")
    args = parser.parse_args()

    bench_path = Path(args.bench_path)
    model_path = Path(args.model)

    if not bench_path.exists():
        raise SystemExit(f"llama-bench not found at {bench_path}")
    if not model_path.exists():
        raise SystemExit(f"model not found at {model_path}")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("this probe is intended for Apple Silicon macOS hosts")

    results = []
    for name, (device, n_gpu_layers) in DEFAULT_CONFIGS:
        results.append(
            run_bench(
                bench_path=bench_path,
                model_path=model_path,
                prompt_tokens=args.prompt_tokens,
                gen_tokens=args.gen_tokens,
                repetitions=args.repetitions,
                flash_attn=not args.no_flash_attn,
                config_name=name,
                device=device,
                n_gpu_layers=n_gpu_layers,
            )
        )

    best = max(results, key=lambda row: row["gen_tps"])

    payload = {
        "system": read_system_profile(),
        "bench_path": str(bench_path),
        "model": str(model_path),
        "prompt_tokens": args.prompt_tokens,
        "gen_tokens": args.gen_tokens,
        "repetitions": args.repetitions,
        "flash_attn": not args.no_flash_attn,
        "results": results,
        "recommended": {
            "name": best["name"],
            "device": best["device"],
            "n_gpu_layers": best["n_gpu_layers"],
            "prompt_tps": best["prompt_tps"],
            "gen_tps": best["gen_tps"],
        },
    }

    if args.output == "json":
        print(json.dumps(payload, indent=2))
        return 0

    print("# Apple Silicon Backend Probe")
    print()
    for key, value in payload["system"].items():
        print(f"- {key}: {value}")
    print(f"- model: {payload['model']}")
    print(f"- prompt_tokens: {payload['prompt_tokens']}")
    print(f"- gen_tokens: {payload['gen_tokens']}")
    print(f"- repetitions: {payload['repetitions']}")
    print(f"- flash_attn: {payload['flash_attn']}")
    print()
    print("| Config | Device | NGL | Prompt TPS | Gen TPS |")
    print("|--------|--------|-----|------------|---------|")
    for row in results:
        print(
            f"| {row['name']} | {row['device']} | {row['n_gpu_layers']} | "
            f"{row['prompt_tps']:.2f} | {row['gen_tps']:.2f} |"
        )
    print()
    print(
        f"Recommended default: `{best['device']}` with `--n-gpu-layers {best['n_gpu_layers']}` "
        f"(gen {best['gen_tps']:.2f} tok/s, prompt {best['prompt_tps']:.2f} tok/s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
