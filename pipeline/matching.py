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


if __name__ == "__main__":
    images, posts = load_data()

    for post in posts:
        result = get_suggestion(post, images)
        print(f"\n=== {post['title']} ===")
        print(json.dumps(result, indent=2))