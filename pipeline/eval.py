import json
from matching import load_data, get_suggestion

EVAL_OUTPUT = "output/eval_results.json"


def run_eval():
    images, posts = load_data()

    # Only score posts that have a real expected image — the two
    # deliberately ambiguous/no-match posts are excluded from precision
    # scoring itself, but we still record what the system did with them.
    scorable_posts = [p for p in posts if p["expected_category"] is not None]
    no_match_posts = [p for p in posts if p["expected_category"] is None]

    correct = 0
    case_results = []

    for post in scorable_posts:
        result = get_suggestion(post, images)
        is_correct = (
            result["result"] == "accepted"
            and next(
                img["category"] for img in images
                if img["filename"] == result["image"]
            ) == post["expected_category"]
        )
        if is_correct:
            correct += 1

        case_results.append({
            "post_id": post["id"],
            "post_title": post["title"],
            "expected_category": post["expected_category"],
            "result": result["result"],
            "matched_image": result.get("image"),
            "correct": is_correct
        })

    precision = round(correct / len(scorable_posts), 4) if scorable_posts else 0.0

    no_match_results = []
    for post in no_match_posts:
        result = get_suggestion(post, images)
        no_match_results.append({
            "post_id": post["id"],
            "post_title": post["title"],
            "result": result["result"],
            "correctly_refused": result["result"] == "no_confident_match"
        })

    summary = {
        "top_1_precision": precision,
        "correct": correct,
        "total_scorable": len(scorable_posts),
        "case_results": case_results,
        "no_match_cases": no_match_results
    }

    with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Top-1 precision: {correct}/{len(scorable_posts)} = {precision:.1%}")
    print(f"\nNo-match cases (should all be 'no_confident_match'):")
    for r in no_match_results:
        status = "✓" if r["correctly_refused"] else "✗"
        print(f"  {status} {r['post_title']}: {r['result']}")


if __name__ == "__main__":
    run_eval()