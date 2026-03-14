#!/usr/bin/env python3

import argparse
import json
import socket
import statistics
import subprocess
import sys
import time
import urllib.request


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
}


WORKLOADS = {
    "pingpong": [
        ("alpha", 0),
        ("beta", 0),
        ("alpha", 1),
        ("beta", 1),
        ("alpha", 2),
        ("beta", 2),
    ],
    "bursty": [
        ("alpha", 0),
        ("beta", 0),
        ("alpha", 1),
        ("alpha", 2),
        ("beta", 1),
        ("beta", 2),
    ],
}


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_json(method: str, url: str, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=120) as resp:
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
            time.sleep(0.2)
    raise RuntimeError(f"server failed to start on port {port}")


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


def parse_case(value: str):
    try:
        label, remainder = value.split("=", 1)
        binary, mode = remainder.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "case must look like label=/path/to/llama-server:mode"
        ) from exc

    if mode not in {"none", "explicit", "continuity"}:
        raise argparse.ArgumentTypeError("mode must be one of: none, explicit, continuity")

    return {
        "label": label,
        "binary": binary,
        "mode": mode,
    }


def start_server(case, args, port: int):
    proc = subprocess.Popen(
        [
            case["binary"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            args.model,
            "--parallel",
            str(args.parallel),
            "--threads",
            str(args.threads),
            "--temp",
            "0",
            "--seed",
            "42",
            "--n-predict",
            str(args.n_predict),
            "--metrics",
            "--no-slots",
            "--no-webui",
            "--no-jinja",
            "--n-gpu-layers",
            str(args.n_gpu_layers),
        ],
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


def run_case(case, args):
    port = find_free_port()
    proc = start_server(case, args, port)
    base = f"http://127.0.0.1:{port}"
    token_by_convo = {}
    histories = {key: [] for key in DEFAULT_CONVOS}
    requests = []
    try:
        for convo_id, turn_idx in WORKLOADS[args.workload]:
            histories[convo_id].append(("User", DEFAULT_CONVOS[convo_id][turn_idx]))
            body = {
                "prompt": build_prompt(args.prefix_memos, histories[convo_id]),
                "cache_prompt": True,
                "n_predict": args.n_predict,
                "temperature": 0,
                "seed": 42,
            }
            headers = {}
            if case["mode"] == "explicit":
                body["metadata"] = {
                    "session_key": convo_id,
                    "lineage_key": convo_id,
                }
            elif case["mode"] == "continuity" and convo_id in token_by_convo:
                headers["X-Neural-Continuity"] = token_by_convo[convo_id]

            t0 = time.perf_counter()
            resp_headers, resp_body = http_json("POST", base + "/completion", body=body, headers=headers)
            latency_ms = (time.perf_counter() - t0) * 1000

            token = resp_headers.get("X-Neural-Continuity")
            if token:
                token_by_convo[convo_id] = token

            timings = resp_body["timings"]
            histories[convo_id].append(("Assistant", resp_body["content"].strip()))
            requests.append(
                {
                    "conversation": convo_id,
                    "turn_index": turn_idx + 1,
                    "latency_ms": latency_ms,
                    "prompt_n": timings["prompt_n"],
                    "prompt_ms": timings.get("prompt_ms", 0.0),
                }
            )

        _, metrics_text = http_json("GET", base + "/metrics")
        metrics = {}
        for line in metrics_text.splitlines():
            if line.startswith("llamacpp:"):
                name, value = line.split()[:2]
                metrics[name.replace("llamacpp:", "")] = float(value)

        return {
            "case": case,
            "requests": requests,
            "metrics": metrics,
        }
    finally:
        stop_server(proc)


def summarize(runs):
    followups = []
    prompt_ms = []
    prompt_n = []
    metric_samples = {}

    for run in runs:
        followup_reqs = run["requests"][2:]
        followups.extend(req["latency_ms"] for req in followup_reqs)
        prompt_ms.extend(req["prompt_ms"] for req in followup_reqs)
        prompt_n.extend(req["prompt_n"] for req in followup_reqs)
        for key, value in run["metrics"].items():
            metric_samples.setdefault(key, []).append(value)

    summary = {
        "median_followup_latency_ms": statistics.median(followups),
        "median_followup_prompt_ms": statistics.median(prompt_ms),
        "median_followup_prompt_n": statistics.median(prompt_n),
    }
    for key, values in metric_samples.items():
        summary[key] = values
    return summary


def main():
    parser = argparse.ArgumentParser(description="Profile session parking and continuity behavior.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--workload", choices=sorted(WORKLOADS.keys()), default="pingpong")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=1)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--prefix-memos", type=int, default=80)
    args = parser.parse_args()

    payload = {"workload": args.workload, "cases": {}}

    for case in args.case:
        runs = []
        for trial in range(args.trials):
            print(f"running {case['label']} trial {trial + 1}/{args.trials}", file=sys.stderr)
            runs.append(run_case(case, args))
        payload["cases"][case["label"]] = {
            "config": case,
            "summary": summarize(runs),
            "runs": runs,
        }

    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps({k: v["summary"] for k, v in payload["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
