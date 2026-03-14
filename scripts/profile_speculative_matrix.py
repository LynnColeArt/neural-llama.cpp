#!/usr/bin/env python3

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request

from profile_session_parking import DEFAULT_CONVOS, WORKLOADS, build_prompt, find_free_port, wait_ready


def parse_spec_case(value: str):
    try:
        label, remainder = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "spec case must look like label=type[,n_max[,n_min]]"
        ) from exc

    parts = remainder.split(",")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError("spec case supports at most type,n_max,n_min")

    spec_type = parts[0]
    if not spec_type:
        raise argparse.ArgumentTypeError("speculative type cannot be empty")

    def parse_optional(index: int, default: int):
        if len(parts) <= index or parts[index] == "":
            return default
        return int(parts[index])

    return {
        "label": label,
        "speculative_type": spec_type,
        "n_max": parse_optional(1, 16),
        "n_min": parse_optional(2, 0),
    }


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


def start_server(args, port: int):
    cmd = [
        args.binary,
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


def run_case(spec_case, args):
    port = find_free_port()
    proc = start_server(args, port)
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
                "speculative.type": spec_case["speculative_type"],
                "speculative.n_max": spec_case["n_max"],
                "speculative.n_min": spec_case["n_min"],
            }
            if spec_case["speculative_type"] != "none":
                if args.ngram_size_n is not None:
                    body["speculative.ngram_size_n"] = args.ngram_size_n
                if args.ngram_size_m is not None:
                    body["speculative.ngram_size_m"] = args.ngram_size_m
                if args.ngram_min_hits is not None:
                    body["speculative.ngram_m_hits"] = args.ngram_min_hits
            headers = {}
            if args.mode == "explicit":
                body["metadata"] = {
                    "session_key": convo_id,
                    "lineage_key": convo_id,
                }
            elif args.mode == "continuity" and convo_id in token_by_convo:
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
                    "predicted_n": timings.get("predicted_n", 0),
                    "predicted_ms": timings.get("predicted_ms", 0.0),
                    "predicted_per_second": timings.get("predicted_per_second", 0.0),
                    "draft_n": timings.get("draft_n", 0),
                    "draft_n_accepted": timings.get("draft_n_accepted", 0),
                }
            )

        _, metrics_text = http_json("GET", base + "/metrics")
        metrics = {}
        for line in metrics_text.splitlines():
            if line.startswith("llamacpp:"):
                name, value = line.split()[:2]
                metrics[name.replace("llamacpp:", "")] = float(value)

        return {
            "spec_case": spec_case,
            "requests": requests,
            "metrics": metrics,
        }
    finally:
        stop_server(proc)


def summarize(runs):
    followups = []
    prompt_ms = []
    prompt_n = []
    predicted_ms = []
    predicted_n = []
    predicted_per_second = []
    draft_n = []
    draft_n_accepted = []
    metric_samples = {}

    for run in runs:
        followup_reqs = run["requests"][2:]
        followups.extend(req["latency_ms"] for req in followup_reqs)
        prompt_ms.extend(req["prompt_ms"] for req in followup_reqs)
        prompt_n.extend(req["prompt_n"] for req in followup_reqs)
        predicted_ms.extend(req["predicted_ms"] for req in followup_reqs)
        predicted_n.extend(req["predicted_n"] for req in followup_reqs)
        predicted_per_second.extend(req["predicted_per_second"] for req in followup_reqs)
        draft_n.extend(req["draft_n"] for req in followup_reqs)
        draft_n_accepted.extend(req["draft_n_accepted"] for req in followup_reqs)
        for key, value in run["metrics"].items():
            metric_samples.setdefault(key, []).append(value)

    summary = {
        "median_followup_latency_ms": statistics.median(followups),
        "median_followup_prompt_ms": statistics.median(prompt_ms),
        "median_followup_prompt_n": statistics.median(prompt_n),
        "median_followup_predicted_ms": statistics.median(predicted_ms),
        "median_followup_predicted_n": statistics.median(predicted_n),
        "median_followup_predicted_per_second": statistics.median(predicted_per_second),
        "median_followup_draft_n": statistics.median(draft_n),
        "median_followup_draft_n_accepted": statistics.median(draft_n_accepted),
    }
    for key, values in metric_samples.items():
        summary[key] = values
    return summary


def main():
    parser = argparse.ArgumentParser(description="Profile speculative decoding modes on a continuity-aware server workload.")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec-case", action="append", type=parse_spec_case, required=True)
    parser.add_argument("--mode", choices=["none", "explicit", "continuity"], default="continuity")
    parser.add_argument("--workload", choices=sorted(WORKLOADS.keys()), default="pingpong")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=1)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--prefix-memos", type=int, default=80)
    parser.add_argument("--ngram-size-n", type=int)
    parser.add_argument("--ngram-size-m", type=int)
    parser.add_argument("--ngram-min-hits", type=int)
    parser.add_argument("--server-arg", action="append", default=[], help="extra llama-server args as flag or flag=value without leading dashes")
    args = parser.parse_args()

    payload = {
        "mode": args.mode,
        "workload": args.workload,
        "spec_cases": {},
    }

    for spec_case in args.spec_case:
        runs = []
        for trial in range(args.trials):
            print(f"running {spec_case['label']} trial {trial + 1}/{args.trials}", file=sys.stderr)
            runs.append(run_case(spec_case, args))
        payload["spec_cases"][spec_case["label"]] = {
            "config": spec_case,
            "summary": summarize(runs),
            "runs": runs,
        }

    ranking = []
    for label, data in payload["spec_cases"].items():
        summary = data["summary"]
        ranking.append({
            "label": label,
            "median_followup_latency_ms": summary["median_followup_latency_ms"],
            "median_followup_prompt_ms": summary["median_followup_prompt_ms"],
            "median_followup_predicted_ms": summary["median_followup_predicted_ms"],
            "median_followup_predicted_per_second": summary["median_followup_predicted_per_second"],
            "median_followup_draft_n": summary["median_followup_draft_n"],
            "median_followup_draft_n_accepted": summary["median_followup_draft_n_accepted"],
            "speculative_acceptance_ratio": summary.get("speculative_acceptance_ratio", []),
        })
    ranking.sort(key=lambda item: item["median_followup_latency_ms"])
    payload["ranking"] = ranking

    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
