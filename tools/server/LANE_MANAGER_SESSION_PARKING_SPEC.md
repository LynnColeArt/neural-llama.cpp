Status: Draft
Date: 2026-03-10

# Lane Manager and Session Parking Spec

## Summary

This document defines an experimental `llama-server` runtime mode for
session-aware lane scheduling.

The target use case is not generic web serving. It is an actor-style local
runtime where:

- many logical sessions may exist at once
- only a few should be hot at the same time
- callers should continue using the ordinary OpenAI-compatible
  `/v1/chat/completions` surface

The first client for this experiment is Qwench, but the runtime should stay
general enough to support future Busy-style local inference clients.

Implementation tracking for this experiment lives in:

- `./LANE_MANAGER_SESSION_PARKING_PROJECT_PLAN.md`

## V1 Decisions

- Resume fidelity target is whole-request continuity only.
- `1`, `2`, and `4` hot lanes are treated as benchmark questions, not constants.
- Idle same-session lanes may stay resident as hot parked state; serialized cold parking is the spill path under lane pressure.
- Parent/child lineage influences priority, not hard affinity.
- Parked session state is process-local in v1.
- External API stays unchanged; scheduler metadata is private.

## Validation Snapshot

Focused validation for the resident hot-parking change was run against the tiny stories test model
(`ggml-org/test-model-stories260K`) with:

- `--n-predict 1`
- `--ctx-size 2048`
- a long shared prefix to make reuse and restore behavior dominate the request
- `5` trials per scenario and `24` requests per trial
- completion latency measured from `/completion` only; `/metrics` was collected once after each scenario so observability overhead did not pollute the timing

Observed results on this branch versus commit `6b5a75b1`:

- `2` slots, same-session hot reuse:
  - median completion latency improved by `6.3%`
  - parked-session restore attempts dropped from `23` to `0`
  - output bytes and `prompt_n` matched baseline across all trials
- `1` slot, forced session displacement:
  - median completion latency improved by `1.8%`
  - restore attempts stayed unchanged at `22`
  - output bytes and `prompt_n` matched baseline across all trials

The implementation intentionally does not keep single-slot sessions resident. That policy preserved the
multi-slot reuse win without introducing a displacement regression in the one-slot case.
- No mid-stream migration or durable parking in v1.

## Problem statement

The current server architecture is shaped around:

- one primary `server_context`
- one shared `llama_context`
- multiple logical slots in that context
- one central `update_slots()` loop

That works well for multi-sequence batching, but it is still fundamentally a
single shared inference engine. It does not map cleanly onto an actor runtime
where parent sessions, child missions, and peer agents should be suspendable
and resumable without each requiring a permanently hot execution context.

## Design goal

Introduce a backend lane manager that:

- keeps the external API stable
- supports a small configurable number of hot execution lanes
- parks inactive logical sessions as restorable sequence state
- restores a parked session into a hot lane when it needs service

This is not a goal to remove contexts entirely.

Contexts remain the execution engine.

The experimental shift is:

- fewer hot contexts
- more parked logical sessions

## Non-goals

- Rewrite callers to understand lane ids.
- Introduce composite multi-lane decoding for one request.
- Implement arbitrary raw KV buffer mutation as a public feature.
- Guarantee mid-token preemption in the first slice.
- Replace router mode.

## External compatibility contract

The experimental runtime must preserve the ordinary request surface:

- `/v1/chat/completions`
- `/v1/responses` if routed through the same completion engine
- normal SSE behavior
- normal tool-call payloads

Any session-affinity hints must be backend-private extensions such as:

- optional HTTP headers
- optional ignored metadata fields

Requests without those hints must still work.

## Core architecture

### Lane manager

A new manager sits between HTTP request handling and active execution.

Responsibilities:

- maintain a pool of hot lanes
- maintain a registry of parked session state
- choose a lane for each incoming logical session
- restore parked state before execution
- park state again after safe yield/completion boundaries

### Hot lane

A hot lane owns:

- a live `llama_context`
- associated per-context compute and output buffers
- lane-local execution state
- the currently bound logical session, if any

### Parked session record

A parked session record stores enough runtime state to resume a session later.

Minimum fields:

- opaque session key
- model identity
- prompt/token lineage metadata
- serialized per-sequence state blob
- restore flags and versioning metadata
- timestamps and recent scheduling metadata

Recommended fields:

- last lane id
- parent/child lineage hint
- priority class
- last successful restore time
- failure counters

## Important design rule

Do not treat this as "raw KV injection."

Prefer supported state and sequence-memory primitives:

- `llama_state_seq_get_size[_ext]()`
- `llama_state_seq_get_data[_ext]()`
- `llama_state_seq_set_data[_ext]()`
- `llama_memory_seq_rm()`
- `llama_memory_seq_cp()`
- `llama_memory_seq_add()`

The runtime may use lower-level details internally, but the architecture should
be built around supported sequence-state operations rather than ad hoc direct
buffer surgery.

## Lane count

Lane count must be configurable.

The first experimental values are:

- `1`
- `2`
- `4`

This runtime should not assume that more lanes are always better. Lane count is
expected to trade off:

- memory
- queueing delay
- restore churn
- backend concurrency limits

## Request identity and private affinity

The runtime needs a stable logical-session key if it is going to park and
restore session state.

Suggested private transport shape:

- `X-Neural-Session-Key`
- `X-Neural-Lineage-Key` optional
- `X-Neural-Priority-Class` optional

These names are illustrative. The main requirement is:

- keep them private
- keep them optional
- avoid turning them into user-facing API contract

## Lifecycle

### 1. Request arrival

An HTTP worker receives a normal completion request.

It extracts:

- model name
- request body
- streaming mode
- private affinity metadata if present

### 2. Session lookup

The lane manager resolves the logical session key.

If no key is present:

- treat the request as stateless or ephemeral
- execute without parked-state restore

If a key is present:

- check whether parked state exists
- check whether the session is already active on another lane

### 3. Lane selection

The lane manager selects a hot lane using:

- same-model compatibility
- current occupancy
- last-lane affinity
- priority class
- fairness policy

One logical session may be active on at most one lane at a time.

### 4. Restore

If the chosen lane is not already bound to the same session:

- clear or sanitize the lane's prior bound session state
- restore the parked sequence state into the lane
- restore any required lane-local metadata

If restore fails:

- fall back to a safe replay/rebuild strategy if possible
- mark the parked record as degraded
- never cross-contaminate sessions

### 5. Run

The lane executes the request normally.

Streaming, tool calls, and ordinary response conversion should remain identical
to the caller.

### 6. Park

At a safe boundary, the lane manager snapshots restorable session state back to
the parked-session registry and releases the lane for future work.

## Safe parking boundaries

The first slice should park only at whole-request boundaries:

- completed non-streaming response
- completed streaming response
- explicit cancellation
- request failure after cleanup

Future slices may consider additional boundaries, but the first slice should
favor correctness over aggressiveness.

## Scheduler policy

The lane scheduler should preserve:

- correctness first
- fairness second
- affinity third

Suggested policy order:

1. do not violate single-active-session rule
2. prefer a free lane already affinity-matched to the session
3. otherwise choose the least costly restore path
4. interactive work may jump ahead of background mission work
5. avoid starvation of parked mission sessions

## Memory model

This design aims to reduce hot memory pressure relative to "one hot context per
logical agent."

Expected memory buckets:

- model weights
- hot lane contexts
- parked session blobs
- scheduler bookkeeping

Memory will still rise with lane count because each hot lane has its own active
context and compute state.

However, this should still be materially cheaper than keeping one hot context
per logical session when many sessions are mostly idle.

## Restore fidelity

Session restore is only useful if it preserves enough state to make generation
feel continuous.

Important note:

Sequence state alone may not be sufficient for perfect fidelity.

Potential state classes:

- sequence/KV state
- sampler state
- grammar/guidance state
- reasoning/tool-call partial parse state if any exists outside the model layer

The first slice may accept less-than-perfect fidelity if it is explicit about
its limitations, but the runtime should not claim seamless resume unless those
other state classes are accounted for.

## Failure semantics

Correctness beats cleverness.

Acceptable degradation:

- replay more prompt than ideal
- lose affinity and reassign to another lane
- drop to a smaller effective lane count

Unacceptable failure:

- emit one session's state into another session
- corrupt streaming envelopes
- continue generation on stale or partially restored state without marking the
  failure

If an aborted decode leaves partial memory state behind, restore/cleanup logic
must respect the documented sequence memory constraints and query current memory
state before reusing the lane.

## Metrics

The experimental runtime should emit metrics for at least:

- configured lane count
- active lane count
- parked session count
- lane queue depth
- restore attempts
- restore failures
- average restore latency
- average queue wait
- average active run time
- affinity hit rate

These metrics are necessary to judge whether `2` or `4` lanes actually improve
the actor-style workload.

## First experimental slice

The first slice should stay intentionally narrow.

### Required

- configurable lane count
- private session-affinity metadata
- parked session registry
- whole-request restore and re-park
- compatibility with current completion and SSE surface

### Deferred

- mid-stream migration
- cross-process parking
- durable on-disk parked state
- full sampler-state fidelity if that blocks the first proof
- adaptive lane-count scaling

## Testing expectations

The experiment should be judged on workloads that resemble actor runtimes, not
just ordinary throughput microbenchmarks.

Minimum useful scenarios:

1. one session, one lane
2. two independent sessions, two lanes
3. parent session parked while child mission runs
4. four-session contention test with mixed short and long turns
5. cancellation during streaming followed by safe reuse of the lane

## Relationship to current server model

This experiment does not replace the current batching/slot model overnight.

It introduces a new execution mode whose philosophy is:

- current server: many live sequences in one shared context
- lane manager mode: a few live contexts, many parked logical sessions

That is a materially different optimization target.

## Canonical stance

The backend should own:

- lane count
- parking
- restore
- affinity
- fairness

The caller should own:

- messages
- tool declarations
- session identity
- mission/orchestrator semantics

That division is what keeps the experiment useful.
