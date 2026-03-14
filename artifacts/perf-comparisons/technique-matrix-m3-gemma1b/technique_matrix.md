# Technique Matrix M3 Max Gemma 1B

- model: `/Users/sam/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf`
- device: `MTL0`
- parallel: `4`
- kv_unified: `True`
- trials: `2`
- rounds: `2`

## Ranking

| case | round wall ms | latency ms | prompt ms | gen tok/s | restore attempts | cache hit | cache admit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `continuity_tokens=off,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=off` | 388.64 | 381.38 | 73.47 | 118.21 | 8 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=on` | 395.11 | 387.59 | 68.51 | 119.13 | 8 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=on` | 396.87 | 389.55 | 72.18 | 119.37 | 8 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=on` | 402.45 | 396.53 | 42.81 | 92.35 | 0 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=on` | 411.27 | 404.05 | 41.01 | 91.97 | 0 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=off` | 422.69 | 418.79 | 47.92 | 91.76 | 0 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=on` | 423.33 | 419.37 | 69.07 | 92.40 | 3 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=off` | 423.53 | 419.69 | 53.92 | 91.60 | 0 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=on` | 424.97 | 420.65 | 54.28 | 92.64 | 4 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=off` | 429.23 | 420.46 | 67.90 | 91.32 | 8 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=off` | 432.81 | 426.61 | 64.09 | 90.95 | 4 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=off` | 434.56 | 427.06 | 48.64 | 91.23 | 4 | 0.000 | 0.000 |

## Technique Summary

| technique | on median round ms | off median round ms | pairwise delta % | wins enabled | losses enabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| `continuity_tokens` | 424.25 | 416.98 | -0.10% | 3 | 3 |
| `hot_resident_sessions` | 423.43 | 395.99 | 7.11% | 0 | 4 |
| `prefer_empty_session_slots` | 416.98 | 424.15 | -1.07% | 4 | 0 |
| `prompt_cache_admission` | 406.86 | 426.38 | -2.64% | 5 | 1 |

## Granularity

- best case: `continuity_tokens=off,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=off`
- worst case: `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=off`
- spread: `11.81%`
- recommendation: `keep_granular`
- reason: the effects vary across combinations enough that a single hard-coded policy would hide material tradeoffs
- keep configurable: `continuity_tokens, hot_resident_sessions, prompt_cache_admission`
