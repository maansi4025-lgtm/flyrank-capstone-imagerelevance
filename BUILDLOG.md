# Build Log — AI usage, honestly

This capstone was built with Claude as a build partner throughout. This log
records where it helped, where its first suggestion was wrong, and what I
changed.

## Where AI genuinely helped
- Scaffolding each phase's code structure (Pydantic schemas, FastAPI routes,
  the batch job loop) — I reviewed and understood every function before
  running it.
- Debugging real, non-obvious failures step by step: a PowerShell column-
  width issue that looked like a missing file, a Docker/venv mix-up, and
  most notably a vision model (moondream) that was silently copying the
  prompt's example text back as its answer instead of describing the image.

## Where the first suggestion was wrong, and what changed
- **moondream → llava.** The first vision model choice (moondream) reliably
  produced structurally valid but content-wrong JSON — it copied the
  example in the prompt almost verbatim, especially on wolf photos. This
  wasn't caught by schema validation, since the output was still
  syntactically correct. I only found it by comparing outputs across
  different images and noticing they were suspiciously identical. Switching
  to `llava` fixed it — confirmed by re-running all 50 images and manually
  checking that captions were genuinely distinct per image.
- **The first similarity threshold (0.5) was too strict.** Real eval runs
  showed correct matches scoring as low as 0.39, so every threshold was
  rejecting genuinely correct answers. I lowered it to 0.35 based on the
  actual score distribution from my own eval set, not a guess.
- **The initial prompt used a filled-in example**, which is what caused the
  model-copying bug above. Removing the concrete example (replacing it with
  a format description) fixed it — confirmed by re-running the batch job
  and comparing outputs.

## What I can explain
I can walk through any part of this codebase: the schema validation and
repair-retry logic in `pipeline/vision.py`, the cosine similarity and guard
logic in `pipeline/matching.py`, and the review workflow in `api/main.py`.