from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from matching import load_data, get_suggestion

app = FastAPI(title="Image Matching Engine")

REVIEWS_FILE = "output/reviews.json"


def load_reviews():
    if os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_reviews(reviews):
    with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)


class ReviewDecision(BaseModel):
    decision: str  # "approved" or "rejected"


@app.get("/")
def root():
    return {
        "name": "AI Image Understanding & Content Matching Engine",
        "endpoints": ["/posts/{post_id}/suggestion", "/posts", "/reviews"]
    }


@app.get("/posts")
def list_posts():
    _, posts = load_data()
    return [{"id": p["id"], "title": p["title"]} for p in posts]


@app.get("/posts/{post_id}/suggestion")
def get_post_suggestion(post_id: str):
    images, posts = load_data()
    post = next((p for p in posts if p["id"] == post_id), None)
    if post is None:
        raise HTTPException(status_code=404, detail={"error": "Post not found"})

    result = get_suggestion(post, images)
    return result


@app.post("/posts/{post_id}/review")
def review_suggestion(post_id: str, decision: ReviewDecision):
    if decision.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail={"error": "decision must be 'approved' or 'rejected'"})

    images, posts = load_data()
    post = next((p for p in posts if p["id"] == post_id), None)
    if post is None:
        raise HTTPException(status_code=404, detail={"error": "Post not found"})

    suggestion = get_suggestion(post, images)

    reviews = load_reviews()
    review_entry = {
        "post_id": post_id,
        "post_title": post["title"],
        "suggested_image": suggestion.get("image"),
        "suggestion_result": suggestion["result"],
        "decision": decision.decision
    }
    reviews.append(review_entry)
    save_reviews(reviews)

    return review_entry


@app.get("/reviews")
def list_reviews():
    return load_reviews()