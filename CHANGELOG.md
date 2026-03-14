# Changelog

This file tracks fork-specific changes that materially affect downstream
integrators such as Qwench. It is not intended to duplicate upstream
`llama.cpp` release notes.

## 2026-03-12

### Automatic continuity tokens for `llama-server`

- Added a first automatic continuity path for chat/completions requests.
- When no explicit scheduler identity is supplied, `llama-server` can now:
  - accept an opaque continuity token from:
    - `X-Neural-Continuity`
    - `metadata.continuity_token`
    - `Cookie: NeuralContinuity=...`
  - synthesize backend session continuity from that token
  - issue a new continuity token when no token is present
  - return the token in:
    - `X-Neural-Continuity`
    - `Set-Cookie: NeuralContinuity=...`
- This begins the transition away from caller-managed `session_key` /
  `lineage_key` metadata for normal interactive clients.

### Why this matters

- Downstream clients such as Qwench manage prompt truth themselves.
- Backend continuity should remain a server-side cache/locality concern, not a
  client contract.
- Reused client-supplied scheduler identity was shown to poison otherwise valid
  post-tool turns; server-issued continuity gives the backend a safer boundary
  for future invalidation and recovery logic.

### Related fork work already present in this repo

- Apple Silicon NPU support for the forked runtime
- experimental lane manager / parked session restore for `llama-server`

Future entries should continue documenting fork-specific behavior changes at
that same boundary: cache continuity, lane scheduling, restore semantics,
runtime APIs, and hardware-specific behavior.
