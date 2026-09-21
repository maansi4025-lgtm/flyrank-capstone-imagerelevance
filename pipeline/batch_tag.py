import os
import json
import time
from datetime import datetime, timezone
from vision import tag_image

IMAGES_DIR = "images"
OUTPUT_FILE = "output/image_tags.json"
LOG_FILE = "output/vision_cost_log.jsonl"


def get_all_images():
    return sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def process_all_images():
    os.makedirs("output", exist_ok=True)
    images = get_all_images()

    results = []
    flagged = []
    total_start = time.time()

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        for i, filename in enumerate(images, start=1):
            image_path = os.path.join(IMAGES_DIR, filename)
            print(f"[{i}/{len(images)}] Processing {filename}...")

            call_start = time.time()
            tags = tag_image(image_path)
            duration_ms = round((time.time() - call_start) * 1000, 1)

            log_entry = {
                "image": filename,
                "duration_ms": duration_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "tagged" if tags else "flagged",
            }
            log.write(json.dumps(log_entry) + "\n")

            if tags is None:
                print(f"  -> FLAGGED (validation failed twice)")
                flagged.append(filename)
                results.append({
                    "filename": filename,
                    "status": "flagged",
                    "tags": None
                })
                continue

            status = "tagged" if tags.confidence >= 0.6 else "flagged"
            if status == "flagged":
                print(f"  -> FLAGGED (low confidence: {tags.confidence})")
                flagged.append(filename)
            else:
                print(f"  -> {tags.category} ({tags.confidence})")

            results.append({
                "filename": filename,
                "status": status,
                "tags": json.loads(tags.model_dump_json())
            })

    total_duration = round(time.time() - total_start, 1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(images)} images processed in {total_duration}s")
    print(f"Tagged: {len(images) - len(flagged)}, Flagged: {len(flagged)}")
    if flagged:
        print(f"Flagged images: {flagged}")


if __name__ == "__main__":
    process_all_images()