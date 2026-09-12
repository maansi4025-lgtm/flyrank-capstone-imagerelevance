# Design — AI Image Understanding & Content Matching Engine

## Problem

Given a small library of images and a set of blog posts, automatically suggest the
correct image for each post based on what the image actually depicts — not
filenames or keywords — while explicitly refusing to suggest a wrong image
(a visually similar but semantically incorrect one, e.g. a wolf on a fox post)
rather than guessing.

## Non-goal

This is not an image search engine and not a general-purpose CMS. It does not
generate images, does not support arbitrary free-text image search, and does
not attempt multi-image-per-post ranking beyond a simple top-N list. One
vision model, one embedding model, one corpus theme (animals).

## Image metadata schema

Every image, after processing, produces this validated structure:

```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "wild", "forest"],
  "caption": "A red fox standing in a forest",
  "confidence": 0.94
}
```

- `subject`: free text, the specific thing depicted (e.g. "red fox", not just "animal")
- `category`: one of a small closed list for this corpus — `["fox", "wolf", "dog", "bear", "deer"]`
- `attributes`: 2-5 short descriptive tags
- `caption`: one sentence, used as the text that gets embedded
- `confidence`: 0.0-1.0, the model's own certainty

Any response missing a field, with a `category` outside the closed list, or with
`confidence` outside 0-1 is rejected and retried once; a second failure marks the
image `status: flagged` rather than accepting a guess.

## Matching strategy

1. Embed every image's `caption` into a vector, store it.
2. Embed every post's body text into a vector, store it.
3. For a given post, rank all images by cosine similarity (post vector vs. image vectors).
4. Pass the top candidate(s) through the mismatch guard before returning anything.

## The mismatch guard — rules, in order

A candidate image is **rejected** if any of these are true:
1. `confidence` on the image's tags was below `0.6` (never surface an uncertain tag as a confident match)
2. The image's `category` is not mentioned anywhere in the post's own extracted subject/category (a fox post's expected category is `fox`; a `wolf`-tagged image fails this even with high similarity)
3. Cosine similarity is below the tuned threshold (initial guess: `0.75`, to be tuned against the eval set in Phase 4)

Every rejection returns a human-readable reason string, e.g.:
`"Animal category mismatch: expected fox, detected wolf"` or
`"Similarity 0.61 below threshold 0.75"`.

If every candidate is rejected, the endpoint returns `"no confident match"` with
the reasons for the top candidate that was closest.

## Database design

- `images` — id, filename, category (nullable until tagged), subject, attributes (array), caption, confidence, status (`pending`/`tagged`/`flagged`), created_at
- `image_embeddings` — image_id (FK), vector, model_name, created_at
- `posts` — id, title, body, category (manually seeded, matches image category list), created_at
- `post_embeddings` — post_id (FK), vector, model_name, created_at
- `suggestions` — id, post_id (FK), image_id (FK), similarity_score, guard_result (`accepted`/`rejected`), reason, created_at
- `reviews` — id, suggestion_id (FK), decision (`approved`/`rejected`), reviewed_at

Indexes: `images.status`, `suggestions.post_id`, `suggestions.guard_result` — the
queries that matter are "give me all tagged images" and "give me this post's
suggestions."

## API surface (sketch, subject to change during build)

- `POST /images/process` — trigger the batch tagging job
- `GET /images` — list images with their tags/status
- `GET /posts/:id/suggestions` — ranked, guard-filtered suggestions for a post
- `POST /suggestions/:id/review` — approve or reject a suggestion
- `GET /costs` — per-call cost log

## Stack

- Python + FastAPI
- PostgreSQL via Docker (plain array column for embeddings at this scale — ~50 images doesn't need pgvector)
- Ollama (local vision model + embedding model) — no API key, no quota
- Pydantic for schema validation throughout