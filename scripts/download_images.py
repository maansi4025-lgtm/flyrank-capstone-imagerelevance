import os
import requests
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.environ["PEXELS_API_KEY"]
HEADERS = {"Authorization": API_KEY}
IMAGES_PER_CATEGORY = 10

CATEGORIES = ["red fox", "wolf", "dog", "bear", "deer"]

OUTPUT_DIR = "images"

def download_category(query: str, category_slug: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    response = requests.get(
        "https://api.pexels.com/v1/search",
        headers=HEADERS,
        params={"query": query, "per_page": IMAGES_PER_CATEGORY},
        timeout=15
    )
    response.raise_for_status()
    data = response.json()

    for i, photo in enumerate(data["photos"], start=1):
        image_url = photo["src"]["medium"]
        filename = f"{category_slug}_{i}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        img_response = requests.get(image_url, timeout=15)
        img_response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(img_response.content)

        print(f"Saved {filename}")


if __name__ == "__main__":
    for category in CATEGORIES:
        slug = category.replace(" ", "-")
        print(f"--- Downloading: {category} ---")
        download_category(category, slug)

    print("Done.")