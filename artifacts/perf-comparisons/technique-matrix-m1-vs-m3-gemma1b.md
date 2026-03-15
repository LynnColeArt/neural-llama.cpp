# Technique Matrix Comparison: M1 Max vs M3 Max

Model:

- `ggml-org_gemma-3-1b-it-Q4_K_M.gguf`

Base server shape on both machines:

- device: `MTL0`
- `parallel=4`
- `kv_unified=true`
- `cache_type_k=f16`
- `cache_type_v=f16`
- `n_gpu_layers=999`
- `n_predict=32`
- `rounds=2`
- `trials=2`

## High-Level Result

Both machines independently recommend `keep_granular`.

- M3 Max spread across combinations: `11.81%`
- M1 Max spread across combinations: `16.06%`

That is too much spread to justify one global hard-coded policy.

## Best Cases

M3 Max best:

- `continuity_tokens=off`
- `hot_resident_sessions=off`
- `prefer_empty_session_slots=off`
- `prompt_cache_admission=off`

M1 Max best:

- `continuity_tokens=off`
- `hot_resident_sessions=on`
- `prefer_empty_session_slots=on`
- `prompt_cache_admission=on`

The best cases diverge materially, which is the strongest argument against a
single global default policy.

## Per-Technique Effects

### `prompt_cache_admission`

This is the most defensible cross-machine default to turn on.

- M3 Max pairwise median delta when enabled: `-2.64%`
- M1 Max pairwise median delta when enabled: `-4.54%`
- wins when enabled:
  - M3 Max: `5` wins / `1` loss
  - M1 Max: `5` wins / `1` loss

Recommendation:

- default `on`
- still keep configurable

### `prefer_empty_session_slots`

This looks safe to fold into the hot-resident policy rather than keep as a
first-class top-level tuning knob.

- M3 Max pairwise median delta when enabled: `-1.07%`
- M1 Max pairwise median delta when enabled: `0.00%`
- wins when enabled:
  - M3 Max: `4` wins / `0` losses
  - M1 Max: `3` wins / `1` loss

Recommendation:

- if `hot_resident_sessions` is enabled, force `prefer_empty_session_slots=on`
- do not treat it as a separate top-level default policy

### `hot_resident_sessions`

This is hardware/workload dependent and should not be globally hard-coded.

- M3 Max pairwise median delta when enabled: `+7.11%`
- M1 Max pairwise median delta when enabled: `+0.56%`

Interpretation:

- on M3 Max, enabling hot residency is clearly harmful on this workload
- on M1 Max, it is near-neutral overall, but the best M1 configuration still
  uses it together with `prefer_empty_session_slots=on` and
  `prompt_cache_admission=on`

Recommendation:

- keep configurable
- consider a machine-aware default:
  - M3 Max default `off`
  - M1 Max default `on` only if paired with `prefer_empty_session_slots=on`
    and `prompt_cache_admission=on`

### `continuity_tokens`

This is the most workload-dependent feature in the set.

- M3 Max pairwise median delta when enabled: `-0.10%`
- M1 Max pairwise median delta when enabled: `-5.20%`

Interpretation:

- basically neutral on M3 Max for this workload
- meaningfully helpful on M1 Max

Recommendation:

- keep configurable
- do not hard-code globally
- reasonable future policy is client-class based rather than hardware-only:
  enable for stateless chat clients, disable for clients already sending stable
  session identity explicitly

## Recommended Policy

1. Hard-code `prefer_empty_session_slots=on` whenever `hot_resident_sessions=on`.
2. Default `prompt_cache_admission=on`.
3. Keep `hot_resident_sessions` configurable and likely hardware-aware.
4. Keep `continuity_tokens` configurable and likely client/workload-aware.

## Practical Next Step

The next implementation step should not be more branch forking. It should be a
small auto-policy layer that:

- always enables `prompt_cache_admission`
- ties `prefer_empty_session_slots` to `hot_resident_sessions`
- leaves `hot_resident_sessions` and `continuity_tokens` as the primary knobs
- optionally chooses the default for `hot_resident_sessions` by machine family
  once more hardware data exists
