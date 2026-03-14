#!/usr/bin/env python3

import argparse
import json
import platform
import re
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

SYSTEM_FIELD_ORDER = (
    "platform",
    "machine",
    "model_name",
    "model_identifier",
    "model_number",
    "chip",
    "cpu_cores",
    "cpu_total_cores",
    "cpu_performance_cores",
    "cpu_efficiency_cores",
    "cpu_other_cores",
    "cpu_physical_cores",
    "cpu_logical_cores",
    "gpu_name",
    "gpu_cores",
    "npu_visible",
    "npu_device_count",
    "npu_visible_devices",
    "npu_core_count",
    "memory",
    "memory_bytes",
    "memory_bytes_usable",
    "unified_memory",
    "metal_support",
    "system_firmware",
    "os_loader_version",
)


def parse_cpu_core_split(value: str) -> dict[str, int]:
    proc_match = re.search(r"proc\s+(?P<total>\d+):(?P<perf>\d+):(?P<eff>\d+):(?P<other>\d+)", value)
    if proc_match:
        result = {
            "cpu_total_cores": int(proc_match.group("total")),
            "cpu_performance_cores": int(proc_match.group("perf")),
            "cpu_efficiency_cores": int(proc_match.group("eff")),
        }
        other = int(proc_match.group("other"))
        if other:
            result["cpu_other_cores"] = other
        return result

    match = re.search(r"(?P<total>\d+)\s+\((?P<perf>\d+)\s+Performance and (?P<eff>\d+)\s+Efficiency\)", value)
    if not match:
        return {}

    return {
        "cpu_total_cores": int(match.group("total")),
        "cpu_performance_cores": int(match.group("perf")),
        "cpu_efficiency_cores": int(match.group("eff")),
    }


def format_cpu_cores(value: object) -> object:
    if not isinstance(value, str):
        return value

    proc_match = re.search(r"proc\s+(?P<total>\d+):(?P<perf>\d+):(?P<eff>\d+):(?P<other>\d+)", value)
    if not proc_match:
        return value

    total = proc_match.group("total")
    perf = proc_match.group("perf")
    eff = proc_match.group("eff")
    other = int(proc_match.group("other"))
    if other:
        return f"{total} ({perf} Performance, {eff} Efficiency, {other} Other)"
    return f"{total} ({perf} Performance and {eff} Efficiency)"


def format_metal_support(value: object) -> object:
    if not isinstance(value, str):
        return value

    match = re.fullmatch(r"spdisplays_metal(\d+)", value)
    if not match:
        return value

    return f"Metal {match.group(1)}"


def read_sysctl_value(name: str) -> str | None:
    try:
        output = subprocess.check_output(["sysctl", "-n", name], text=True)
    except Exception:
        return None
    return output.strip()


def read_ane_profile() -> dict[str, object]:
    try:
        output = subprocess.check_output(["ioreg", "-l"], text=True, errors="ignore")
    except Exception:
        return {}

    visible_devices = sorted(set(re.findall(r"\bane(\d+)@", output)))
    if not visible_devices:
        return {
            "npu_visible": False,
            "npu_device_count": 0,
        }

    return {
        "npu_visible": True,
        "npu_device_count": len(visible_devices),
        "npu_visible_devices": [f"ane{device_id}" for device_id in visible_devices],
        "npu_core_count": "not_exposed_by_public_macos_apis",
    }


def read_system_profile() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    try:
        payload = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"],
            text=True,
        )
        data = json.loads(payload)
    except Exception:
        info.update(read_ane_profile())
        return info

    hardware = next(iter(data.get("SPHardwareDataType", [])), {})
    graphics = next(iter(data.get("SPDisplaysDataType", [])), {})

    info.update(
        {
            "model_name": hardware.get("machine_name"),
            "model_identifier": hardware.get("machine_model"),
            "model_number": hardware.get("model_number"),
            "chip": hardware.get("chip_type"),
            "cpu_cores": format_cpu_cores(hardware.get("number_processors")),
            "memory": hardware.get("physical_memory"),
            "system_firmware": hardware.get("boot_rom_version"),
            "os_loader_version": hardware.get("os_loader_version"),
            "gpu_name": graphics.get("sppci_model"),
            "gpu_cores": graphics.get("sppci_cores"),
            "metal_support": format_metal_support(graphics.get("spdisplays_mtlgpufamilysupport")),
            "unified_memory": platform.machine() == "arm64",
        }
    )

    cpu_cores = info.get("cpu_cores")
    if isinstance(cpu_cores, str):
        info.update(parse_cpu_core_split(cpu_cores))

    for sysctl_key, field_name in (
        ("hw.memsize", "memory_bytes"),
        ("hw.memsize_usable", "memory_bytes_usable"),
        ("hw.physicalcpu", "cpu_physical_cores"),
        ("hw.logicalcpu", "cpu_logical_cores"),
    ):
        value = read_sysctl_value(sysctl_key)
        if value is not None:
            try:
                info[field_name] = int(value)
            except ValueError:
                info[field_name] = value

    info.update(read_ane_profile())

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
    for key in SYSTEM_FIELD_ORDER:
        if key not in payload["system"]:
            continue
        value = payload["system"][key]
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
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
