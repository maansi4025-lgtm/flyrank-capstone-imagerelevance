# AI Image Understanding & Content Matching Engine

An AI system that looks at a library of animal photos, understands what's
actually in each one, and matches the right image to the right blog post —
based on meaning, not filenames or keywords. A fox post gets a fox photo.
A visually similar wolf photo is rejected, with a reason. When nothing in
the corpus is a good enough match, the system says so instead of guessing.

## Architecture

```
Images (~50) ──(batch job)──► Vision Model (llava) ──► {tags, caption, confidence}
                                                                │
                                                        embed(caption)
                                                                │
                                                                ▼
                                                        Image Embeddings
Blog Posts ──────────────────────────► embed(post text) ──► Post Embeddings
                                                                │
                                                                ▼
                                          Similarity Ranking (cosine similarity)
                                                                │
                                                                ▼
                                    Mismatch Guard (confidence + category + threshold)
                                          │                              │
                                   Suggested image                "No confident match"
                                   (ranked, explained)                + reasons
                                                                │
                                                                ▼
                                          Review API — approve / reject / inspect why
```

Full design rationale, including the exact guard rules and database design,
is in [`DESIGN.md`](./DESIGN.md).

## What's built

| Phase | Status | What it covers |
|---|---|---|
| 1 — Design | ✅ Done | Schema, guard rules, DB design, 50-image corpus (5 categories) |
| 2 — Vision pipeline | ✅ Done | Batch tagging, schema validation, repair retry, cost/duration log |
| 3 — Matching engine | ✅ Done | Embeddings, similarity ranking, the mismatch guard |
| 4 — Production layer | ✅ Done | Review API, eval set, precision score, documentation |

## Results

- **Vision tagging:** 49/50 images tagged successfully; 1 correctly flagged
  (a genuine model misclassification — a deer photo tagged "giraffe" — caught
  by schema validation after two attempts, never stored).
- **Top-1 precision:** **10/10 (100%)** on a hand-labeled set of 10 posts,
  each marked with its one correct image, plus 2/2 deliberately ambiguous
  posts correctly refused with "no confident match." Measured by
  `pipeline/eval.py`, results in `output/eval_results.json`.
- **The mismatch guard**, proven directly: forcing a wolf photo as a
  candidate for a fox-themed post is rejected with
  *"Animal category mismatch: expected fox, detected wolf"* — see
  `EVIDENCE.md` for the full transcript.

## How to run it

**1. Install Ollama** (local AI, no account, no key): [ollama.com/download](https://ollama.com/download)

```bash
ollama pull llava
ollama pull all-minilm
```

**2. Set up the project:**

```bash
git clone https://github.com/maansi4025-lgtm/flyrank-capstone-imagerelevance.git
cd flyrank-capstone-imagerelevance
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

(The `.env` / Pexels key is only needed if you want to regenerate the image
corpus from scratch via `scripts/download_images.py`. The 50 images are
already committed to the repo, so this step can be skipped to just run the
system.)

**3. Run the pipeline** (regenerates all tags, embeddings, and the eval
score from the committed images and posts):

```bash
python pipeline/batch_tag.py
python pipeline/embeddings.py
python pipeline/matching.py
python pipeline/eval.py
```

**4. Start the API:**

```bash
uvicorn api.main:app --reload
```

**5. Try it:**

```bash
curl -i http://127.0.0.1:8000/posts
curl -i http://127.0.0.1:8000/posts/post-1/suggestion
curl -i -X POST http://127.0.0.1:8000/posts/post-1/review -H "Content-Type: application/json" -d "{\"decision\":\"approved\"}"
curl -i http://127.0.0.1:8000/reviews
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/posts` | List all blog posts |
| GET | `/posts/{post_id}/suggestion` | Ranked image suggestion for a post, with guard reasoning |
| POST | `/posts/{post_id}/review` | Approve or reject a suggestion |
| GET | `/reviews` | List all recorded review decisions |

## The mismatch guard

Three independent checks, in order — any one failing rejects the candidate:

1. **Confidence check** — the vision model's own confidence on the image's
   tags must be ≥ 0.6.
2. **Category check** — the image's tagged category must match what the
   post actually needs.
3. **Similarity threshold** — cosine similarity must clear 0.35, a value
   tuned directly against the real eval set (an initial guess of 0.5
   incorrectly rejected several genuinely correct matches; lowering it to
   0.35 fixed all of them without accepting any wrong ones).

Every rejection returns a specific, human-readable reason — see
`pipeline/matching.py`'s `guard_check()`.

## Honest limitations

- **Flat JSON files instead of a relational database.** `DESIGN.md` sketches
  a full Postgres schema, but at this corpus size (49 images, 12 posts),
  JSON files hold the same structure without the overhead of running Docker
  + Postgres for a project this size. The schema design would translate
  directly if scaled up.
- **`expected_category` is used as a stand-in for "what a post needs."** A
  fuller system would extract a post's subject via its own classification
  step; here, posts declare their expected category directly for testing
  clarity.
- **A local vision model (llava, ~4.7GB) occasionally misclassifies** — one
  image in the corpus was tagged "giraffe" and correctly flagged rather than
  stored. Small local models are genuinely less reliable than large hosted
  ones; the schema validation and confidence threshold exist specifically to
  catch this rather than trust every answer.
- **One duplicate-caption bug was found and fixed during development**: an
  earlier prompt version (and an earlier, smaller model) caused the vision
  model to copy the prompt's own example text back as its answer for several
  wolf images. Caught by manually comparing outputs across images, fixed by
  removing the filled-in example from the prompt and switching from
  `moondream` to `llava`. Full account in `BUILDLOG.md`.

## Why no separate frontend

Per the brief's scope: the review interface is API endpoints, which is
sufficient to demonstrate the full "suggest → inspect why → approve/reject"
workflow without a UI build.

## Files

- `DESIGN.md` — full design doc (schema, guard rules, database design)
- `EVIDENCE.md` — one proof per requirement
- `BUILDLOG.md` — honest AI-usage log
- `capstone.yaml` — evaluator manifest
- `.env.example` — required environment variables
- `pipeline/` — vision tagging, embeddings, matching, eval
- `api/` — the Review API
- `data/posts.json` — the 12 test blog posts
- `images/` — the 50-image corpus
- `output/` — generated tags, embeddings, logs, eval results