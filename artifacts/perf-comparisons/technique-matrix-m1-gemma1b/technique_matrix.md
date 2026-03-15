# Technique Matrix technique-matrix-m1-gemma1b

- model: `/Users/sam/Library/Caches/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf`
- device: `MTL0`
- parallel: `4`
- kv_unified: `True`
- trials: `2`
- rounds: `2`

## Ranking

| case | round wall ms | latency ms | prompt ms | gen tok/s | restore attempts | cache hit | cache admit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=on` | 500.14 | 494.29 | 64.83 | 89.90 | 0 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=off` | 501.48 | 496.78 | 65.16 | 77.98 | 4 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=on` | 501.74 | 491.81 | 89.39 | 96.06 | 8 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=off` | 503.61 | 493.33 | 102.72 | 95.76 | 8 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=on` | 505.92 | 501.06 | 101.35 | 97.25 | 0 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=off` | 537.50 | 536.53 | 96.89 | 97.71 | 0 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=off` | 543.82 | 538.82 | 82.16 | 87.07 | 0 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=on` | 547.01 | 538.09 | 86.74 | 72.69 | 4 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=on` | 552.76 | 537.55 | 113.73 | 69.58 | 8 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=off,prefer_empty_session_slots=off,prompt_cache_admission=off` | 571.67 | 564.16 | 123.62 | 69.15 | 8 | 0.000 | 0.000 |
| `continuity_tokens=on,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=on` | 573.87 | 568.17 | 127.59 | 69.86 | 4 | 0.000 | 0.000 |
| `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=off` | 580.48 | 570.88 | 74.41 | 69.21 | 3 | 0.000 | 0.000 |

## Technique Summary

| technique | on median round ms | off median round ms | pairwise delta % | wins enabled | losses enabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| `continuity_tokens` | 504.76 | 549.89 | -5.20% | 4 | 2 |
| `hot_resident_sessions` | 540.66 | 528.18 | 0.56% | 2 | 2 |
| `prefer_empty_session_slots` | 521.71 | 549.89 | 0.00% | 3 | 1 |
| `prompt_cache_admission` | 526.47 | 540.66 | -4.54% | 5 | 1 |

## Granularity

- best case: `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=on,prompt_cache_admission=on`
- worst case: `continuity_tokens=off,hot_resident_sessions=on,prefer_empty_session_slots=off,prompt_cache_admission=off`
- spread: `16.06%`
- recommendation: `keep_granular`
- reason: the effects vary across combinations enough that a single hard-coded policy would hide material tradeoffs
- keep configurable: `continuity_tokens, hot_resident_sessions, prefer_empty_session_slots, prompt_cache_admission`
