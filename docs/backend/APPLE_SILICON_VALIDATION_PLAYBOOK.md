# Apple Silicon Validation Playbook

This playbook exists to answer one question with evidence:

> Does this fork choose and use the right accelerated backend on the current
> Apple Silicon machine, and can we prove that choice is better than the
> alternatives?

The immediate next target is the original M1 Max test machine, but the protocol
is intentionally generic so it can be reused on any Apple Silicon host.

## What We Are Trying To Prove

We are not trying to prove that one backend name is always best.

We are trying to prove all of the following:

1. the machine exposes the expected Apple Silicon acceleration devices
2. `auto` chooses the correct default backend policy on that machine
3. explicit `MTL0` and explicit `COREML0` are both measurable
4. full-offload policy (`--n-gpu-layers 999`) still behaves correctly
5. the chosen default is justified by throughput, not assumption
6. the startup logs and memory breakdown agree with the benchmark story

## Current Hypothesis

On the local M3 Max validation run used to shape this branch:

- `MTL0 --n-gpu-layers 999` beat `COREML0 --n-gpu-layers 999`
- `auto` needed to prefer Metal by default to match the fastest path
- `COREML0` remained useful as an explicit probe target, but not as a safe
  universal default

The M1 Max may behave differently. That is why this playbook exists.

## Required Inputs

- this branch checked out on the target machine
- Xcode Command Line Tools
- `cmake` and `ninja`
- one locally available benchmark model

The reference model used on the M3 Max was:

- `gemma-3-1b-it-Q4_K_M.gguf`
- expected cache path:
  - `~/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf`

If that file is not already present on the M1 Max, fetch it first using any
preferred method before running the validation capture script.

## Build

From repo root:

```sh
cmake -S . -B build-apple-silicon \
  -DGGML_METAL=ON \
  -DGGML_COREML=ON \
  -DLLAMA_CURL=OFF

cmake --build build-apple-silicon --config Release -j 12 \
  --target llama-server llama-bench
```

## One-Command Capture

From repo root on the target machine:

```sh
./scripts/run_apple_silicon_validation.sh \
  --model "$HOME/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
```

This writes a timestamped artifact directory under:

```text
artifacts/apple-silicon-validation/<hostname>-<timestamp>/
```

## What The Capture Produces

- `system_profile.txt`
  - sanitized raw hardware/display profile from macOS
- `sw_vers.txt`
  - macOS version context
- `git_rev.txt`
  - exact repo commit
- `server_list_devices.txt`
  - what `llama-server --list-devices` reports
- `probe.md`
  - human-readable backend matrix summary
- `probe.json`
  - machine-readable backend matrix summary
- `server_tuning.md`
  - human-readable server tuning matrix covering backend policy, `n_parallel`,
    `kv_unified`, and KV cache-type choices on a real concurrent workload
- `server_tuning.json`
  - machine-readable server tuning matrix
- `profile_technique_matrix.py`
  - optional profiler for sweeping continuity, hot-resident parking, empty-slot
    preference, and prompt-cache admission in every meaningful combination on
    top of a fixed server configuration
- `summary.md`
  - distilled verdict on device exposure, best explicit backend, auto-vs-best
    delta, whether `COREML0` still routes through `MTL0` buffers, the best
    measured server configuration, prompt-cache behavior, and the structured
    machine profile
- `bench_auto.json`
  - the current default (`auto`) benchmark result
- `startup_auto.log`
  - startup/load/offload log for default device selection
- `startup_mtl0.log`
  - startup/load/offload log for explicit Metal
- `startup_coreml0.log`
  - startup/load/offload log for explicit CoreML

## What To Check

### 0. Summary first

Open `summary.md` before digging through the raw artifacts.

It should answer, at a glance:

- what machine was tested
- the full machine profile: model identifier, CPU split, GPU, ANE visibility,
  unified memory, memory size, firmware
- which explicit backend won
- whether `auto` tracked that winning backend closely
- whether explicit `COREML0` still ends up allocating KV/compute buffers on
  `MTL0`
- which measured server configuration won on the concurrent chat workload
- whether prompt-cache admission and hot-session reuse were actually helping

Then use the remaining artifacts to validate or challenge that summary.

Note:

- macOS does not expose a stable public ANE core-count field in the same clean
  way it exposes CPU and GPU cores. The capture records ANE visibility and
  visible ANE devices, and marks NPU core count as not publicly exposed when
  that remains true.

### 1. Device exposure

In `server_list_devices.txt`, confirm whether the machine reports:

- `MTL0`
- `COREML0`

If `COREML0` is missing on the M1 Max, that is immediately important.

### 2. Default backend choice

In `bench_auto.json`, check:

- `devices`
- `backends`
- prompt throughput
- generation throughput

Then compare against `probe.json`.

Success condition:

- `auto` should match or closely track the best explicit backend configuration
  from the probe matrix

Failure condition:

- `auto` lands materially below the best explicit config on the same machine

### 3. Explicit backend comparison

In `probe.md` or `probe.json`, compare at minimum:

- `metal_full999`
- `coreml_full999`

This is the most important pair.

Questions to answer:

- Is M1 Max still a case where `COREML0` wins?
- Or does M1 Max also prefer `MTL0` for this model?
- Is the difference large enough to justify a machine-family rule?

### 4. Partial vs full offload

Still in `probe.md`:

- compare `partial20` vs `full999` for each device

Success condition:

- `full999` should either win or clearly explain why it does not

This matters because the old benchmark lore was built around `999` being the
right “let the runtime clamp” probe.

### 5. Startup and memory story

Inspect:

- `startup_auto.log`
- `startup_mtl0.log`
- `startup_coreml0.log`

You want to confirm:

- which device the model says it is using
- where layers were actually offloaded
- where KV and compute buffers landed
- whether `COREML0` still routes through `MTL0`

If the benchmark says one thing and the startup logs say another, that is a
bug or at least a documentation mismatch.

### 6. Server tuning

Inspect:

- `server_tuning.md`
- `server_tuning.json`

This layer answers a different question than `llama-bench`:

- on this machine, which server configuration actually wins on a concurrent
  continuity-heavy workload?
- does `n_parallel = 4` still make sense?
- does `kv_unified` help or hurt here?
- do alternate KV cache types improve throughput enough to justify them?
- is prompt-cache admission earning its keep, or just creating churn?

The important outputs to compare are:

- `median_followup_round_wall_ms`
- `median_followup_latency_ms`
- `median_followup_prompt_ms`
- `median_followup_predicted_per_second`
- `prompt_cache_hit_ratio`
- `prompt_cache_admission_ratio`
- `scheduler_restore_attempts_total`

### 7. Technique combinations

Once you know the best base server shape from `server_tuning.md`, use
`scripts/profile_technique_matrix.py`
to sweep the higher-level techniques in combination instead of comparing
branches by hand.

Recommended flow:

- lock in one base config first: device, `parallel`, `kv_unified`, `cache_type_k`,
  `cache_type_v`
- run the technique matrix on that base config
- compare `median_followup_round_wall_ms` first, then latency, restore churn,
  and prompt-cache hit/admission ratios
- use the profiler's consolidated technique summary to see whether each flag
  has a consistent effect when enabled, and whether the measured spread is
  small enough to justify hard-coding
- only promote a technique combination if it wins on the metrics that matter

## What To Commit Back

If the M1 Max run completes cleanly, commit the generated artifact directory.

Recommended commit shape:

```text
artifacts/apple-silicon-validation/<hostname>-<timestamp>/
```

Recommended commit message:

```text
apple: add m1 max backend validation artifacts
```

That gives the follow-up work an artifact trail instead of a verbal report.

## What Counts As A Useful Result

Useful results include all of these cases:

- M1 Max confirms Metal should be the generic default
- M1 Max disproves that and shows CoreML is still the better default there
- M1 Max shows both are close, meaning we should prefer a probe-driven policy
- M1 Max exposes a startup/logging mismatch where device labeling is misleading

All of those outcomes are useful because they narrow what “works across any
Apple Silicon” actually needs to mean.

## Optional Enrichment

If you want more operational color during the run:

- watch Activity Monitor GPU history during `metal_full999`
- compare thermals / throttling behavior between `MTL0` and `COREML0`

Those observations are useful, but they are secondary. The primary proof path
is still the recorded throughput plus startup/offload logs.
