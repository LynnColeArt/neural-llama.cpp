#!/usr/bin/env python3

import argparse
import concurrent.futures
import itertools
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


TECHNIQUES = (
    ("continuity_tokens", False),
    ("hot_resident_sessions", True),
    ("prefer_empty_session_slots", True),
    ("prompt_cache_admission", True),
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
        parsed = json.loads(raw) if "application/json" in resp.headers.get("Content-Type", "") else raw
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
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", str(args.model),
        "--parallel", str(args.parallel),
        "--threads", str(args.threads),
        "--temp", "0",
        "--seed", "42",
        "--n-predict", str(args.n_predict),
        "--metrics",
        "--slots",
        "--no-webui",
        "--no-jinja",
        "--n-gpu-layers", str(args.n_gpu_layers),
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
    ]
    if args.device != "auto":
        cmd.extend(["--device", args.device])
    if args.kv_unified:
        cmd.append("--kv-unified")
    cmd.append("--continuity-tokens" if case["continuity_tokens"] else "--no-continuity-tokens")
    cmd.append("--hot-resident-sessions" if case["hot_resident_sessions"] else "--no-hot-resident-sessions")
    cmd.append("--prefer-empty-session-slots" if case["prefer_empty_session_slots"] else "--no-prefer-empty-session-slots")
    cmd.append("--prompt-cache-admission" if case["prompt_cache_admission"] else "--no-prompt-cache-admission")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_ready(port)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)


def completion_request(base: str, convo_id: str, prompt: str, args, continuity_state: dict, use_continuity: bool):
    headers = {}
    body = {
        "prompt": prompt,
        "cache_prompt": True,
        "n_predict": args.n_predict,
        "temperature": 0,
        "seed": 42,
    }
    if use_continuity:
        token = continuity_state.get("token")
        cookie = continuity_state.get("cookie")
        if token:
            headers["X-Neural-Continuity"] = token
        if cookie:
            headers["Cookie"] = cookie
    else:
        body["metadata"] = {
            "session_key": convo_id,
            "lineage_key": convo_id,
            "request_class": "chat",
            "priority_class": "interactive",
        }

    t0 = time.perf_counter()
    resp_headers, resp_body = http_json("POST", base + "/completion", body=body, headers=headers)
    latency_ms = (time.perf_counter() - t0) * 1000

    if use_continuity:
        token = resp_headers.get("X-Neural-Continuity", continuity_state.get("token"))
        cookie = resp_headers.get("Set-Cookie", "")
        if token:
            continuity_state["token"] = token
        if cookie:
            continuity_state["cookie"] = cookie.split(";", 1)[0]
        if not continuity_state.get("token"):
            raise RuntimeError(f"continuity enabled but no token returned for {convo_id}")

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
    continuity_state = {key: {} for key in DEFAULT_CONVOS}
    requests = []
    try:
        for convo_id, turns in DEFAULT_CONVOS.items():
            histories[convo_id].append(("User", turns[0]))
            latency_ms, resp_body = completion_request(
                base,
                convo_id,
                build_prompt(args.prefix_memos, histories[convo_id]),
                args,
                continuity_state[convo_id],
                case["continuity_tokens"],
            )
            histories[convo_id].append(("Assistant", resp_body["content"].strip()))
            requests.append({
                "round": 0,
                "conversation": convo_id,
                "latency_ms": latency_ms,
                "timings": resp_body["timings"],
            })

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
                    executor.submit(
                        completion_request,
                        base,
                        convo_id,
                        prompts[convo_id],
                        args,
                        continuity_state[convo_id],
                        case["continuity_tokens"],
                    ): convo_id
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
                requests.append({
                    "round": round_index,
                    "conversation": convo_id,
                    "latency_ms": results[convo_id]["latency_ms"],
                    "round_wall_ms": round_wall_ms,
                    "timings": resp_body["timings"],
                })

        return {"case": case, "requests": requests, "metrics": collect_metrics(base)}
    finally:
        stop_server(proc)


def summarize(runs: list[dict]) -> dict:
    followup_latencies = []
    round_wall = []
    prompt_ms = []
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
            predicted_per_second.append(timings.get("predicted_per_second", 0.0))

        for key, value in run["metrics"].items():
            metric_samples.setdefault(key, []).append(value)

    summary = {
        "median_followup_round_wall_ms": statistics.median(round_wall),
        "median_followup_latency_ms": statistics.median(followup_latencies),
        "median_followup_prompt_ms": statistics.median(prompt_ms),
        "median_followup_predicted_per_second": statistics.median(predicted_per_second),
    }
    for key, values in metric_samples.items():
        summary[key] = values
    return summary


def render_markdown(payload: dict) -> str:
    lines = [
        "# Technique Matrix",
        "",
        f"- model: `{payload['model']}`",
        f"- device: `{payload['device']}`",
        f"- parallel: `{payload['parallel']}`",
        f"- kv_unified: `{payload['kv_unified']}`",
        f"- cache_type_k: `{payload['cache_type_k']}`",
        f"- cache_type_v: `{payload['cache_type_v']}`",
        "",
        "## Ranking",
        "",
        "| case | continuity | hot resident | empty slot | prompt cache admission | round wall ms | latency ms | prompt ms | gen tok/s | restore attempts | cache hit | cache admit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["ranking"]:
        case = payload["cases"][item["label"]]
        summary = case["summary"]
        lines.append(
            f"| `{item['label']}` | "
            f"{'on' if case['config']['continuity_tokens'] else 'off'} | "
            f"{'on' if case['config']['hot_resident_sessions'] else 'off'} | "
            f"{'on' if case['config']['prefer_empty_session_slots'] else 'off'} | "
            f"{'on' if case['config']['prompt_cache_admission'] else 'off'} | "
            f"{summary['median_followup_round_wall_ms']:.2f} | "
            f"{summary['median_followup_latency_ms']:.2f} | "
            f"{summary['median_followup_prompt_ms']:.2f} | "
            f"{summary['median_followup_predicted_per_second']:.2f} | "
            f"{summary.get('scheduler_restore_attempts_total', [0.0])[-1]:.0f} | "
            f"{summary.get('prompt_cache_hit_ratio', [0.0])[-1]:.3f} | "
            f"{summary.get('prompt_cache_admission_ratio', [0.0])[-1]:.3f} |"
        )
    return "\n".join(lines) + "\n"


def build_cases():
    names = [name for name, _ in TECHNIQUES]
    defaults = {name: value for name, value in TECHNIQUES}
    cases = []
    for values in itertools.product((False, True), repeat=len(TECHNIQUES)):
        case = dict(zip(names, values))
        if not case["hot_resident_sessions"]:
            case["prefer_empty_session_slots"] = False
        label = ",".join(f"{name}={'on' if case[name] else 'off'}" for name in names)
        cases.append({
            "label": label,
            **defaults,
            **case,
        })
    dedup = {}
    for case in cases:
        dedup[case["label"]] = case
    return list(dedup.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile every combination of continuity, parking, and prompt-cache policy techniques.")
    parser.add_argument("--server-bin", default="build-apple-silicon/bin/llama-server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", choices=("json", "md"), default="md")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--prefix-memos", type=int, default=180)
    parser.add_argument("--device", default="MTL0")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--kv-unified", action="store_true")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    args = parser.parse_args()

    args.server_bin = Path(args.server_bin)
    args.model = Path(args.model)
    if not args.server_bin.exists():
        raise SystemExit(f"llama-server not found at {args.server_bin}")
    if not args.model.exists():
        raise SystemExit(f"model not found at {args.model}")

    payload = {
        "model": str(args.model),
        "device": args.device,
        "parallel": args.parallel,
        "kv_unified": args.kv_unified,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "trials": args.trials,
        "rounds": args.rounds,
        "cases": {},
        "errors": {},
    }

    for case in build_cases():
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
        ranking.append({
            "label": label,
            "median_followup_round_wall_ms": data["summary"]["median_followup_round_wall_ms"],
            "median_followup_latency_ms": data["summary"]["median_followup_latency_ms"],
        })
    ranking.sort(key=lambda item: (item["median_followup_round_wall_ms"], item["median_followup_latency_ms"]))
    payload["ranking"] = ranking
    if not ranking:
        raise SystemExit(f"no technique-matrix cases completed successfully: {payload['errors']}")

    if args.output == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
