# neural-llama.cpp Fork - Agent Notes

This fork accepts human-written, AI-assisted, automated, and predominantly AI-generated contributions.
Agents are welcome here.

Changes are judged on correctness, clarity, tests, maintainability, and contributor accountability, not on whether a model helped produce them.

## What Matters Here

- Keep changes focused and surgical.
- Preserve public APIs unless the task explicitly requires changing them.
- Do not add dependencies or new subsystems casually.
- Match nearby file patterns before introducing new structure.

## Project Facts That Are Easy To Miss

- This fork carries Apple Silicon NPU/CoreML work; read `docs/backend/COREML.md` before changing that path.
- `llama-server` internals and active runtime experiments are documented in `tools/server/README-dev.md`.
- `libllama` and `/v1/*` behaviors are compatibility-sensitive; behavior changes need tests and docs.
- Keep runtime failures visible; do not hide backend or parsing problems behind silent fallbacks.

## Definition Of Done

- Relevant tests or focused validation were run and reported.
- Behavior changes include nearby doc or test updates.
- Diffs stay narrow; avoid opportunistic refactors.
- The submitter can explain and maintain the change.

## When Uncertain

- Read the nearest implementation and follow local conventions.
- Prefer the smallest correct change over a clever one.
- Pause before changing public API, scheduler behavior, or backend memory semantics.
