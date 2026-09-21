import ollama
import json
import os

EMBED_MODEL = "all-minilm"


def embed_text(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def embed_all_images():
    with open("output/image_tags.json", "r", encoding="utf-8") as f:
        images = json.load(f)

    image_embeddings = []
    for img in images:
        if img["status"] != "tagged":
            continue  # skip flagged images — no reliable caption to embed

        caption = img["tags"]["caption"]
        vector = embed_text(caption)
        image_embeddings.append({
            "filename": img["filename"],
            "category": img["tags"]["category"],
            "confidence": img["tags"]["confidence"],
            "caption": caption,
            "embedding": vector
        })
        print(f"Embedded {img['filename']}")

    os.makedirs("output", exist_ok=True)
    with open("output/image_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(image_embeddings, f)

    print(f"\nEmbedded {len(image_embeddings)} images")


def embed_all_posts():
    with open("data/posts.json", "r", encoding="utf-8") as f:
        posts = json.load(f)

    post_embeddings = []
    for post in posts:
        text = f"{post['title']}. {post['body']}"
        vector = embed_text(text)
        post_embeddings.append({
            "id": post["id"],
            "title": post["title"],
            "expected_category": post["expected_category"],
            "embedding": vector
        })
        print(f"Embedded {post['id']}: {post['title']}")

    with open("output/post_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(post_embeddings, f)

    print(f"\nEmbedded {len(post_embeddings)} posts")


if __name__ == "__main__":
    embed_all_images()
    embed_all_posts()