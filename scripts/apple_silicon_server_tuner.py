#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


DEFAULT_CONVOS = {
    "alpha": [
        "Design a migration from cron jobs to an event worker with safety and rollback details.",
        "Add metrics and a staged canary plan.",
        "Add a replay-protection strategy and incident response note.",
    ],
    "beta": [
        "Design a local-first assistant runtime for shell tasks and screenshots.",
        "Add durable-memory boundaries and failure handling.",
        "Add observability for latency and tool reliability.",
    ],
    "gamma": [
        "Design a search indexing pipeline for mixed PDF and web content with retries.",
        "Add backpressure controls and a corruption recovery path.",
        "Add cost controls and audit logging.",
    ],
    "delta": [
        "Design a collaborative coding agent runtime with checkpointed task execution.",
        "Add failure isolation and partial-result recovery.",
        "Add profiling and rollout safeguards.",
    ],
}


DEFAULT_CASES = (
    {"label": "auto_default", "device": "auto", "parallel": 4, "kv_unified": True, "cache_type_k": "f16", "cache_type_v": "f16"},
    {"label": "metal_p2_kvu1_f16_f16", "device": "MTL0", "parallel": 2, "kv_unified": True, "cache_type_k": "f16", "cache_type_v": "f16"},
    {"label": "metal_p4_kvu1_f16_f16", "device": "MTL0", "parallel": 4, "kv_unified": True, "cache_type_k": "f16", "cache_type_v": "f16"},
    {"label": "metal_p4_kvu0_f16_f16", "device": "MTL0", "parallel": 4, "kv_unified": False, "cache_type_k": "f16", "cache_type_v": "f16"},
    {"label": "metal_p4_kvu1_q8_f16", "device": "MTL0", "parallel": 4, "kv_unified": True, "cache_type_k": "q8_0", "cache_type_v": "f16"},
    {"label": "coreml_p4_kvu1_f16_f16", "device": "COREML0", "parallel": 4, "kv_unified": True, "cache_type_k": "f16", "cache_type_v": "f16"},
)


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_prompt(prefix_memos: int, history) -> str:
    prefix = (
        "System: You are a concise assistant preserving context accurately for long planning conversations.\n"
        "Long notes: "
        + " ".join(f"memo{i}" for i in range(prefix_memos))
        + "\n"
    )
    parts = [prefix]
    for role, text in history:
        parts.append(f"{role}: {text}\n")
    parts.append("Assistant:")
    return "".join(parts)


def http_json(method: str, url: str, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=240) as resp:
        raw = resp.read().decode()
        if "application/json" in resp.headers.get("Content-Type", ""):
            parsed = json.loads(raw)
        else:
            parsed = raw
        return dict(resp.headers.items()), parsed


def wait_ready(port: int) -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            _, body = http_json("GET", f"http://127.0.0.1:{port}/health")
            if body.get("status") == "ok":
                return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"server failed to start on port {port}")


def start_server(case: dict, args, port: int):
    cmd = [
        str(args.server_bin),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(args.model),
        "--parallel",
        str(case["parallel"]),
        "--threads",
        str(args.threads),
        "--temp",
        "0",
        "--seed",
        "42",
        "--n-predict",
        str(args.n_predict),
        "--metrics",
        "--slots",
        "--no-webui",
        "--no-jinja",
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "-ctk",
        case["cache_type_k"],
        "-ctv",
        case["cache_type_v"],
    ]
    if case["device"] != "auto":
        cmd.extend(["--device", case["device"]])
    if case["kv_unified"]:
        cmd.append("--kv-unified")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wait_ready(port)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


def completion_request(base: str, convo_id: str, prompt: str, args):
    body = {
        "prompt": prompt,
        "cache_prompt": True,
        "n_predict": args.n_predict,
        "temperature": 0,
        "seed": 42,
        "metadata": {
            "session_key": convo_id,
            "lineage_key": convo_id,
        },
    }
    t0 = time.perf_counter()
    _, resp_body = http_json("POST", base + "/completion", body=body)
    latency_ms = (time.perf_counter() - t0) * 1000
    return latency_ms, resp_body


def collect_metrics(base: str) -> dict[str, float]:
    _, metrics_text = http_json("GET", base + "/metrics")
    metrics = {}
    for line in metrics_text.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split()[:2]
            metrics[name.replace("llamacpp:", "")] = float(value)
    return metrics


def run_case(case: dict, args) -> dict:
    port = find_free_port()
    proc = start_server(case, args, port)
    base = f"http://127.0.0.1:{port}"
    histories = {key: [] for key in DEFAULT_CONVOS}
    requests = []
    try:
        for convo_id, turns in DEFAULT_CONVOS.items():
            histories[convo_id].append(("User", turns[0]))
            latency_ms, resp_body = completion_request(
                base,
                convo_id,
                build_prompt(args.prefix_memos, histories[convo_id]),
                args,
            )
            histories[convo_id].append(("Assistant", resp_body["content"].strip()))
            requests.append(
                {
                    "round": 0,
                    "conversation": convo_id,
                    "latency_ms": latency_ms,
                    "timings": resp_body["timings"],
                }
            )

        for round_index in range(1, args.rounds + 1):
            prompts = {}
            for convo_id, turns in DEFAULT_CONVOS.items():
                user_turn = turns[min(round_index, len(turns) - 1)]
                histories[convo_id].append(("User", user_turn))
                prompts[convo_id] = build_prompt(args.prefix_memos, histories[convo_id])

            started = time.perf_counter()
            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(DEFAULT_CONVOS)) as executor:
                futures = {
                    executor.submit(completion_request, base, convo_id, prompts[convo_id], args): convo_id
                    for convo_id in DEFAULT_CONVOS
                }
                for future in concurrent.futures.as_completed(futures):
                    convo_id = futures[future]
                    latency_ms, resp_body = future.result()
                    results[convo_id] = {
                        "latency_ms": latency_ms,
                        "body": resp_body,
                    }

            round_wall_ms = (time.perf_counter() - started) * 1000
            for convo_id in DEFAULT_CONVOS:
                resp_body = results[convo_id]["body"]
                histories[convo_id].append(("Assistant", resp_body["content"].strip()))
                requests.append(
                    {
                        "round": round_index,
                        "conversation": convo_id,
                        "latency_ms": results[convo_id]["latency_ms"],
                        "round_wall_ms": round_wall_ms,
                        "timings": resp_body["timings"],
                    }
                )

        return {
            "case": case,
            "requests": requests,
            "metrics": collect_metrics(base),
        }
    finally:
        stop_server(proc)


def summarize(runs: list[dict]) -> dict:
    followup_latencies = []
    round_wall = []
    prompt_ms = []
    prompt_n = []
    predicted_ms = []
    predicted_per_second = []
    metric_samples: dict[str, list[float]] = {}

    for run in runs:
        for req in run["requests"]:
            if req["round"] == 0:
                continue
            followup_latencies.append(req["latency_ms"])
            round_wall.append(req["round_wall_ms"])
            timings = req["timings"]
            prompt_ms.append(timings.get("prompt_ms", 0.0))
            prompt_n.append(timings.get("prompt_n", 0))
            predicted_ms.append(timings.get("predicted_ms", 0.0))
            predicted_per_second.append(timings.get("predicted_per_second", 0.0))

        for key, value in run["metrics"].items():
            metric_samples.setdefault(key, []).append(value)

    summary = {
        "median_followup_round_wall_ms": statistics.median(round_wall),
        "median_followup_latency_ms": statistics.median(followup_latencies),
        "median_followup_prompt_ms": statistics.median(prompt_ms),
        "median_followup_prompt_n": statistics.median(prompt_n),
        "median_followup_predicted_ms": statistics.median(predicted_ms),
        "median_followup_predicted_per_second": statistics.median(predicted_per_second),
    }
    for key, values in metric_samples.items():
        summary[key] = values
    return summary


def render_markdown(payload: dict) -> str:
    lines = [
        "# Apple Silicon Server Tuning",
        "",
        f"- model: `{payload['model']}`",
        f"- rounds: `{payload['rounds']}`",
        f"- trials: `{payload['trials']}`",
        f"- n_predict: `{payload['n_predict']}`",
        f"- n_gpu_layers: `{payload['n_gpu_layers']}`",
        "",
        "## Ranking",
        "",
        "| case | round wall ms | latency ms | prompt ms | gen tok/s | prompt cache hit | prompt cache admit | restore attempts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in payload["ranking"]:
        summary = payload["cases"][item["label"]]["summary"]
        hit_ratio = summary.get("prompt_cache_hit_ratio", [0.0])[-1]
        admit_ratio = summary.get("prompt_cache_admission_ratio", [0.0])[-1]
        restore_attempts = summary.get("scheduler_restore_attempts_total", [0.0])[-1]
        lines.append(
            f"| `{item['label']}` | {summary['median_followup_round_wall_ms']:.2f} | {summary['median_followup_latency_ms']:.2f} | {summary['median_followup_prompt_ms']:.2f} | {summary['median_followup_predicted_per_second']:.2f} | {hit_ratio:.3f} | {admit_ratio:.3f} | {restore_attempts:.0f} |"
        )

    best = payload["ranking"][0]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- best case: `{best['label']}`",
            f"- median follow-up round wall time: `{best['median_followup_round_wall_ms']:.2f} ms`",
            f"- median follow-up latency: `{best['median_followup_latency_ms']:.2f} ms`",
        ]
    )
    if payload.get("errors"):
        lines.extend(["", "## Failed Cases", ""])
        for label, error in sorted(payload["errors"].items()):
            lines.append(f"- `{label}`: {error}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune llama-server concurrency settings on Apple Silicon.")
    parser.add_argument("--server-bin", default="build-apple-silicon/bin/llama-server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", choices=("json", "md"), default="md")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--prefix-memos", type=int, default=180)
    args = parser.parse_args()

    if sys.platform != "darwin":
        raise SystemExit("this tuner is intended for macOS hosts")

    args.server_bin = Path(args.server_bin)
    args.model = Path(args.model)
    if not args.server_bin.exists():
        raise SystemExit(f"llama-server not found at {args.server_bin}")
    if not args.model.exists():
        raise SystemExit(f"model not found at {args.model}")

    payload = {
        "model": str(args.model),
        "trials": args.trials,
        "rounds": args.rounds,
        "n_predict": args.n_predict,
        "n_gpu_layers": args.n_gpu_layers,
        "cases": {},
        "errors": {},
    }

    for case in DEFAULT_CASES:
        try:
            runs = []
            for trial in range(args.trials):
                print(f"running {case['label']} trial {trial + 1}/{args.trials}", file=sys.stderr)
                runs.append(run_case(case, args))
            payload["cases"][case["label"]] = {
                "config": case,
                "summary": summarize(runs),
                "runs": runs,
            }
        except Exception as exc:
            payload["errors"][case["label"]] = str(exc)

    ranking = []
    for label, data in payload["cases"].items():
        ranking.append(
            {
                "label": label,
                "median_followup_round_wall_ms": data["summary"]["median_followup_round_wall_ms"],
                "median_followup_latency_ms": data["summary"]["median_followup_latency_ms"],
            }
        )
    ranking.sort(key=lambda item: (item["median_followup_round_wall_ms"], item["median_followup_latency_ms"]))
    payload["ranking"] = ranking
    if not ranking:
        raise SystemExit(f"no tuning cases completed successfully: {payload['errors']}")

    if args.output == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
