Status: Draft
Date: 2026-03-10

# Lane Manager and Session Parking Project Plan

## Summary

This document is the implementation plan, roadmap, and backlog for the
lane-manager/session-parking experiment in `llama-server`.

The goal is to make `llama-server` usable as an actor-oriented local inference
engine for clients such as Qwench without changing the ordinary
OpenAI-compatible request surface.

This plan should be read alongside:

- `LANE_MANAGER_SESSION_PARKING_SPEC.md`


## Problem

The current server model is optimized for:

- one primary `server_context`
- one shared `llama_context`
- multiple live slots in that context
- one central `update_slots()` loop

That is effective for multi-sequence batching, but it is a poor fit for an
actor-style workload where:

- many logical sessions may exist at once
- only a few should be hot at the same time
- parent sessions and child missions should suspend and resume cheaply

## Project Goal

Introduce a lane manager that keeps a small configurable number of hot
execution lanes while parking inactive logical sessions as restorable state.

The server should remain externally compatible with the current completion
surface while gaining better behavior for multi-agent, mission-heavy local
workloads.

## V1 Decisions

These are locked for the first prototype.

### Fidelity target

V1 targets whole-request resume fidelity, not arbitrary mid-stream fidelity.

A restored session must be semantically faithful from the next request
boundary forward.

Required preserved state for v1:

- sequence/KV state
- sampler RNG state
- active grammar or structured-output state that affects legal next-token space
- stop-condition state when it changes continuation behavior

Explicitly out of scope for v1:

- perfect mid-stream identity fidelity
- speculative-decoding leftovers
- transient debug or tracing state

### Parking scope

Parked session state is process-local only in v1.

It is an engine artifact, not durable user data.

### Lineage policy

Parent/child lineage affects scheduler priority first.

Lineage may provide a soft affinity hint, but it must not become a hard lane
binding rule.

### Scheduler order

The v1 scheduler order is:

1. interactive priority
2. fairness / starvation prevention
3. affinity bonus
4. simple tie-break such as round-robin or least-recently-used

### Transport and normalization

The external request surface stays unchanged.

Any private affinity/session metadata must be normalized into one internal
scheduler request object rather than letting the engine reason directly about
raw HTTP headers throughout the codebase.

Suggested fields:

- `session_key`
- `lineage_key`
- `request_class`
- `priority_class`
- `affinity_hint`
- `interruptibility`
- `source_kind`

## Benchmark Questions

These should be answered with measurement, not debate.

- Is `2` or `4` the better practical default on target local hardware?
- Does `4` materially improve tail latency or throughput under realistic mixed
  workloads?
- How much does affinity bonus help real parent/child mission trees?
- What is the restore overhead under mixed foreground/background contention?

## Explicit Defers

These are intentionally not part of v1.

- mid-token migration or preemption
- durable parked sessions
- cross-process restore
- adaptive lane-count scaling
- scheduler cleverness beyond a simple explainable scoring model

## Workstreams

### Implementation Status (2026-03-10)

- ✅ Scheduler request metadata normalization exists at the HTTP boundary (`X-Neural-*` + `metadata`).
- ✅ Session-aware slot affinity fields are tracked in the active slot state.
- ✅ Non-child duplicate in-session top-level tasks are deferred when the session is already active.
- ✅ Front-of-queue submission remains available for foreground/interactive requests.
- ✅ V1 decisions block added to the spec and tracked in plan.
- ⏳ Full parked-session registry and restore/re-park path are still pending.

### 1. Spec lock

Goal:
make the first slice narrow enough to finish.

Deliverables:

- spec updated with v1 decisions
- this project plan kept in sync with the spec
- explicit benchmark/defer split

### 2. Scheduler request normalization

Goal:
establish one typed internal request object for lane scheduling decisions.

Deliverables:

- private metadata extraction at the HTTP boundary
- typed scheduler request object
- compatibility behavior for requests with no session metadata

### 3. Parked-session registry

Goal:
track process-local parked state safely.

Deliverables:

- parked-session record shape
- model/version compatibility guards
- degraded/failed restore markers
- cleanup rules for unusable parked state

### 4. Hot-lane lifecycle

Goal:
create a configurable pool of hot lanes that can restore, run, park, and
release sessions cleanly.

Deliverables:

- lane count configuration for `1`, `2`, and `4`
- lane bind/unbind lifecycle
- whole-request restore and re-park flow
- cleanup path after cancellation or failed decode

### 5. Scheduler policy

Goal:
make lane choice explainable and measurable.

Deliverables:

- interactive-first priority handling
- waiting-age or starvation-floor logic
- soft affinity bonus
- simple tie-break rule
- single-active-session guarantee

### 6. Instrumentation

Goal:
make scheduler behavior visible enough to tune with evidence.

Required metrics:

- configured lane count
- active lane count
- parked session count
- lane queue depth
- restore attempts
- restore failures
- restore latency
- queue wait time
- active run time
- affinity hit rate
- starvation escapes
- session class distribution when available

### 7. Benchmark and evaluation

Goal:
judge whether the experiment is worthwhile under actor-style workloads.

Minimum scenarios:

1. one session on one lane
2. two independent sessions on two lanes
3. parent session parked while one child mission runs
4. four-session contention with mixed short and long turns
5. long stream plus short urgent interruption

## Roadmap

### Phase 0 - Decision lock and observability design

Outcomes:

- v1 decisions are explicit
- required metrics are fixed
- benchmark scenarios are fixed

### Phase 1 - Request identity and parked-state registry

Outcomes:

- typed scheduler request object exists
- process-local parked-session registry exists
- compatibility guards exist for restore eligibility

### Phase 2 - Single-lane correctness

Outcomes:

- one-lane restore/re-park path works safely
- whole-request resume is correct
- failures degrade toward correctness instead of session contamination

### Phase 3 - Multi-lane scheduling

Outcomes:

- configurable `1`, `2`, and `4` lane counts
- scheduler respects priority, fairness, and affinity
- mission-heavy workloads stop feeling like one serialized queue

### Phase 4 - Benchmark and tuning

Outcomes:

- mixed-workload measurements exist
- restore cost and memory growth are understood
- recommended default lane count is evidence-backed

### Phase 5 - Client validation

Outcomes:

- Qwench remains lane-oblivious
- current completion and SSE behavior remain compatible
- downstream clients have a credible basis for adopting the pattern

## Task List

### Immediate

- ✅ Add a v1 decisions block to `LANE_MANAGER_SESSION_PARKING_SPEC.md`.
- ✅ Define the internal scheduler request object and required fields.
- ⏳ Define the parked-session record shape and restore eligibility checks.
- ⏳ Define the scheduler score inputs and tie-break behavior.
- ⏳ Define the required runtime metrics surface.

### First implementation slice

- Add lane count configuration for `1`, `2`, and `4`.
- Add process-local parked-session registry plumbing.
- Add whole-request restore and whole-request re-park flow.
- Add safe lane cleanup after cancellation and failed restore.
- Add single-active-session enforcement.

### First evaluation slice

- Add mixed-workload benchmark scenarios.
- Measure time to first token, total latency, queue wait, restore overhead, and
  peak memory.
- Compare `1`, `2`, and `4` on target local hardware.
- Record whether `4` is useful, merely possible, or actively harmful.

## Backlog

### Ready after first correctness slice

- tune affinity bonus from real benchmark traces
- refine fairness thresholds
- document restore-fidelity limitations more precisely
- write a short evaluation memo for the chosen default lane count

### Deferred

- durable parked-session format
- cross-process restore
- adaptive lane scaling
- mid-token preemption
- richer priority taxonomy beyond the minimal v1 scheduler constitution

## Acceptance Gates

The first prototype is successful only if:

1. The ordinary completion surface remains externally compatible.
2. The runtime supports configurable lane counts of `1`, `2`, and `4`.
3. Whole-request resume is semantically faithful enough for normal use.
4. Parent/child mission contention improves relative to the one-engine path.
5. Memory growth stays materially below "one hot context per logical session."
6. Streaming and tool-call behavior remain compatible with the current local
   adapter path.

## Canonical Reminder

Every unresolved question should be resolved as one of:

- hard decision now
- benchmark question
- explicit defer

Do not let this experiment drift into elegant uncertainty.
