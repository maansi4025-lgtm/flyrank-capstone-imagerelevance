import json
import math

SIMILARITY_THRESHOLD = 0.35
CONFIDENCE_THRESHOLD = 0.6


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def load_data():
    with open("output/image_embeddings.json", "r", encoding="utf-8") as f:
        images = json.load(f)
    with open("output/post_embeddings.json", "r", encoding="utf-8") as f:
        posts = json.load(f)
    return images, posts


def guard_check(image: dict, similarity: float, expected_category: str = None) -> dict:
    """The mismatch guard — three independent checks."""

    if image["confidence"] < CONFIDENCE_THRESHOLD:
        return {
            "accepted": False,
            "reason": f"Confidence too low ({image['confidence']:.2f}) to trust this tag"
        }

    if expected_category and image["category"] != expected_category:
        return {
            "accepted": False,
            "reason": f"Animal category mismatch: expected {expected_category}, detected {image['category']}"
        }

    if similarity < SIMILARITY_THRESHOLD:
        return {
            "accepted": False,
            "reason": f"Similarity {similarity:.2f} below threshold {SIMILARITY_THRESHOLD}"
        }

    return {"accepted": True, "reason": None}


def rank_images_for_post(post: dict, images: list[dict], top_n: int = 5) -> list[dict]:
    scored = []
    for img in images:
        similarity = cosine_similarity(post["embedding"], img["embedding"])
        scored.append({
            "filename": img["filename"],
            "category": img["category"],
            "caption": img["caption"],
            "confidence": img["confidence"],
            "similarity": round(similarity, 4)
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_n]


def get_suggestion(post: dict, images: list[dict]) -> dict:
    """The full pipeline: rank, then run every candidate through the guard
    until one is accepted, or none are."""

    ranked = rank_images_for_post(post, images, top_n=len(images))

    for candidate in ranked:
        guard_result = guard_check(
            candidate,
            candidate["similarity"],
            expected_category=post.get("expected_category")
        )
        if guard_result["accepted"]:
            return {
                "post_id": post["id"],
                "post_title": post["title"],
                "result": "accepted",
                "image": candidate["filename"],
                "similarity": candidate["similarity"],
                "top_5_candidates": ranked[:5]
            }

    # nothing passed the guard
    best = ranked[0] if ranked else None
    return {
        "post_id": post["id"],
        "post_title": post["title"],
        "result": "no_confident_match",
        "closest_candidate": best["filename"] if best else None,
        "reason": guard_check(
            best, best["similarity"], post.get("expected_category")
        )["reason"] if best else "No images available",
        "top_5_candidates": ranked[:5]
    }


def force_candidate_test(post_id: str, forced_filename: str, images: list[dict], posts: list[dict]):
    """Force a specific image as the only candidate, to directly test the guard
    against a known mismatch — e.g. forcing a wolf photo onto a fox post."""
    post = next(p for p in posts if p["id"] == post_id)
    image = next(img for img in images if img["filename"] == forced_filename)

    similarity = cosine_similarity(post["embedding"], image["embedding"])
    result = guard_check(image, similarity, expected_category=post.get("expected_category"))

    print(f"\n=== FORCED CANDIDATE TEST ===")
    print(f"Post: \"{post['title']}\" (expects: {post['expected_category']})")
    print(f"Forced candidate: {forced_filename} (actual category: {image['category']})")
    print(f"Similarity: {similarity:.4f}")
    print(f"Guard result: {'ACCEPTED' if result['accepted'] else 'REJECTED'}")
    if result["reason"]:
        print(f"Reason: {result['reason']}")


if __name__ == "__main__":
    images, posts = load_data()

    for post in posts:
        result = get_suggestion(post, images)
        print(f"\n=== {post['title']} ===")
        print(f"Result: {result['result']}", end="")
        if result["result"] == "accepted":
            print(f" -> {result['image']} (similarity {result['similarity']})")
        else:
            print(f" -> {result.get('reason')}")

    # The centerpiece demo: force a wolf photo onto the fox post
    force_candidate_test("post-1", "wolf_1.jpg", images, posts)