#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import sys
import time
import urllib.request

from profile_session_parking import build_prompt, find_free_port, wait_ready


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


def parse_case(value: str):
    try:
        label, remainder = value.split("=", 1)
        binary, mode = remainder.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "case must look like label=/path/to/llama-server:mode"
        ) from exc

    if mode not in {"explicit", "continuity"}:
        raise argparse.ArgumentTypeError("mode must be one of: explicit, continuity")

    return {
        "label": label,
        "binary": binary,
        "mode": mode,
    }


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


def start_server(case, args, port: int):
    cmd = [
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
    ]
    for extra in args.server_arg:
        cmd.extend(["--" + extra.split("=", 1)[0], extra.split("=", 1)[1]] if "=" in extra else ["--" + extra])

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


def completion_request(base: str, convo_id: str, prompt: str, args, case, token_by_convo):
    body = {
        "prompt": prompt,
        "cache_prompt": True,
        "n_predict": args.n_predict,
        "temperature": 0,
        "seed": 42,
        "speculative.type": "none" if args.disable_spec else args.speculative_type,
        "speculative.n_max": args.speculative_n_max,
        "speculative.n_min": args.speculative_n_min,
    }
    headers = {}
    if case["mode"] == "explicit":
        body["metadata"] = {
            "session_key": convo_id,
            "lineage_key": convo_id,
        }
    elif convo_id in token_by_convo:
        headers["X-Neural-Continuity"] = token_by_convo[convo_id]

    t0 = time.perf_counter()
    resp_headers, resp_body = http_json("POST", base + "/completion", body=body, headers=headers)
    latency_ms = (time.perf_counter() - t0) * 1000
    return latency_ms, resp_headers, resp_body


def collect_metrics(base: str):
    _, metrics_text = http_json("GET", base + "/metrics")
    metrics = {}
    for line in metrics_text.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split()[:2]
            metrics[name.replace("llamacpp:", "")] = float(value)
    return metrics


def run_case(case, args):
    port = find_free_port()
    proc = start_server(case, args, port)
    base = f"http://127.0.0.1:{port}"
    token_by_convo = {}
    histories = {key: [] for key in DEFAULT_CONVOS}
    rounds = []
    try:
        for convo_id, turns in DEFAULT_CONVOS.items():
            histories[convo_id].append(("User", turns[0]))
            latency_ms, resp_headers, resp_body = completion_request(
                base,
                convo_id,
                build_prompt(args.prefix_memos, histories[convo_id]),
                args,
                case,
                token_by_convo,
            )
            token = resp_headers.get("X-Neural-Continuity")
            if token:
                token_by_convo[convo_id] = token
            histories[convo_id].append(("Assistant", resp_body["content"].strip()))
            rounds.append(
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(
                        completion_request,
                        base,
                        convo_id,
                        prompts[convo_id],
                        args,
                        case,
                        token_by_convo,
                    ): convo_id
                    for convo_id in DEFAULT_CONVOS
                }
                for future in concurrent.futures.as_completed(futures):
                    convo_id = futures[future]
                    latency_ms, resp_headers, resp_body = future.result()
                    results[convo_id] = {
                        "latency_ms": latency_ms,
                        "headers": resp_headers,
                        "body": resp_body,
                    }

            round_wall_ms = (time.perf_counter() - started) * 1000
            for convo_id in DEFAULT_CONVOS:
                resp_headers = results[convo_id]["headers"]
                resp_body = results[convo_id]["body"]
                token = resp_headers.get("X-Neural-Continuity")
                if token:
                    token_by_convo[convo_id] = token
                histories[convo_id].append(("Assistant", resp_body["content"].strip()))
                rounds.append(
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
            "rounds": rounds,
            "metrics": collect_metrics(base),
        }
    finally:
        stop_server(proc)


def summarize(runs):
    followups = []
    followup_round_wall = []
    prompt_ms = []
    prompt_n = []
    predicted_ms = []
    predicted_per_second = []
    draft_n = []
    draft_n_accepted = []
    metric_samples = {}

    for run in runs:
        for req in run["rounds"]:
            if req["round"] == 0:
                continue
            followups.append(req["latency_ms"])
            followup_round_wall.append(req["round_wall_ms"])
            timings = req["timings"]
            prompt_ms.append(timings.get("prompt_ms", 0.0))
            prompt_n.append(timings.get("prompt_n", 0))
            predicted_ms.append(timings.get("predicted_ms", 0.0))
            predicted_per_second.append(timings.get("predicted_per_second", 0.0))
            draft_n.append(timings.get("draft_n", 0))
            draft_n_accepted.append(timings.get("draft_n_accepted", 0))
        for key, value in run["metrics"].items():
            metric_samples.setdefault(key, []).append(value)

    summary = {
        "median_followup_latency_ms": statistics.median(followups),
        "median_followup_round_wall_ms": statistics.median(followup_round_wall),
        "median_followup_prompt_ms": statistics.median(prompt_ms),
        "median_followup_prompt_n": statistics.median(prompt_n),
        "median_followup_predicted_ms": statistics.median(predicted_ms),
        "median_followup_predicted_per_second": statistics.median(predicted_per_second),
        "median_followup_draft_n": statistics.median(draft_n),
        "median_followup_draft_n_accepted": statistics.median(draft_n_accepted),
    }
    draft_total = sum(draft_n)
    accepted_total = sum(draft_n_accepted)
    summary["overall_acceptance_ratio"] = (accepted_total / draft_total) if draft_total else 0.0

    for key, values in metric_samples.items():
        summary[key] = values
    return summary


def main():
    parser = argparse.ArgumentParser(description="Profile speculative decoding under concurrent generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2, help="number of concurrent follow-up rounds after warmup")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--prefix-memos", type=int, default=180)
    parser.add_argument("--speculative-type", default="none")
    parser.add_argument("--speculative-n-max", type=int, default=8)
    parser.add_argument("--speculative-n-min", type=int, default=1)
    parser.add_argument("--disable-spec", action="store_true")
    parser.add_argument("--server-arg", action="append", default=[], help="extra llama-server args as flag or flag=value without leading dashes")
    args = parser.parse_args()

    payload = {
        "rounds": args.rounds,
        "speculative_type": args.speculative_type,
        "cases": {},
    }

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

    ranking = []
    for label, data in payload["cases"].items():
        ranking.append(
            {
                "label": label,
                "median_followup_round_wall_ms": data["summary"]["median_followup_round_wall_ms"],
                "median_followup_latency_ms": data["summary"]["median_followup_latency_ms"],
                "overall_acceptance_ratio": data["summary"]["overall_acceptance_ratio"],
            }
        )
    ranking.sort(key=lambda item: item["median_followup_round_wall_ms"])
    payload["ranking"] = ranking

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
