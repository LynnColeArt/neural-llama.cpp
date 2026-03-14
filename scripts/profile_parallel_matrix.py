#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_profile(script_path: Path, args, parallel_value: int):
    with tempfile.NamedTemporaryFile(prefix=f"parallel-{parallel_value}-", suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    cmd = [
        sys.executable,
        str(script_path),
        "--model",
        args.model,
        "--output",
        str(output_path),
        "--workload",
        args.workload,
        "--trials",
        str(args.trials),
        "--parallel",
        str(parallel_value),
        "--threads",
        str(args.threads),
        "--n-predict",
        str(args.n_predict),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--prefix-memos",
        str(args.prefix_memos),
    ]
    for case in args.case:
        cmd.extend(["--case", case])

    print(f"profiling parallel={parallel_value}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    with output_path.open() as handle:
        payload = json.load(handle)
    output_path.unlink(missing_ok=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Compare llama-server session-parking behavior across --parallel values.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", required=True, help="Pass through to profile_session_parking.py")
    parser.add_argument("--parallel", action="append", type=int, required=True, dest="parallel_values")
    parser.add_argument("--workload", default="pingpong", choices=["pingpong", "bursty"])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=1)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--prefix-memos", type=int, default=80)
    args = parser.parse_args()

    script_path = Path(__file__).with_name("profile_session_parking.py")
    matrix = {
        "model": args.model,
        "workload": args.workload,
        "parallel_values": args.parallel_values,
        "cases": {},
    }

    for parallel_value in args.parallel_values:
        payload = run_profile(script_path, args, parallel_value)
        matrix["cases"][str(parallel_value)] = payload["cases"]

    best = {}
    for case_spec in args.case:
        label = case_spec.split("=", 1)[0]
        ranked = []
        for parallel_value in args.parallel_values:
            summary = matrix["cases"][str(parallel_value)][label]["summary"]
            ranked.append({
                "parallel": parallel_value,
                "median_followup_latency_ms": summary["median_followup_latency_ms"],
                "median_followup_prompt_ms": summary["median_followup_prompt_ms"],
                "scheduler_restore_attempts_total": summary.get("scheduler_restore_attempts_total", []),
                "prompt_cache_hit_ratio": summary.get("prompt_cache_hit_ratio", []),
            })
        ranked.sort(key=lambda item: item["median_followup_latency_ms"])
        best[label] = ranked

    matrix["ranking"] = best

    with open(args.output, "w") as handle:
        json.dump(matrix, handle, indent=2)

    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
