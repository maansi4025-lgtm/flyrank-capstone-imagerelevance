import ollama
import json
import re
from pydantic import BaseModel, ValidationError, confloat
from typing import Literal, Optional

Category = Literal["fox", "wolf", "dog", "bear", "deer"]

class ImageTags(BaseModel):
    subject: str
    category: Category
    attributes: list[str]
    caption: str
    confidence: confloat(ge=0.0, le=1.0)


PROMPT = """You are looking at a photo of an animal. Respond with ONLY a JSON object describing it — no other text before or after.

Use this exact format, filled in with your own real observations of THIS image (this is an example of the shape, not the answer):

{"subject": "gray wolf", "category": "wolf", "attributes": ["thick fur", "alert", "forest"], "caption": "A gray wolf standing among trees", "confidence": 0.9}

Rules:
- "category" must be exactly one of: fox, wolf, dog, bear, deer
- "attributes" is a list of 2 to 4 short words describing what you actually see
- "confidence" is a number between 0.0 and 1.0
- Output ONLY the JSON object on one line. No explanation, no markdown, no extra text."""


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def tag_image(image_path: str) -> Optional[ImageTags]:
    response = ollama.chat(
        model="moondream",
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [image_path]
        }],
        options={"num_predict": 300, "temperature": 0.1}
    )
    raw_text = response["message"]["content"]

    try:
        data = extract_json(raw_text)
        return ImageTags(**data)
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        print(f"  Validation failed: {e}")
        print(f"  Raw response: {raw_text[:300]}")
        return None
def tag_image(image_path: str, retry_with_error: str = None) -> Optional[ImageTags]:
    prompt = PROMPT
    if retry_with_error:
        prompt = PROMPT + f"\n\nYour previous answer was rejected for this reason: {retry_with_error}\nReturn only corrected JSON matching the format above, with 'attributes' as short descriptive words (not numbers)."

    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }],
        options={"num_predict": 300, "temperature": 0.1}
    )
    raw_text = response["message"]["content"]
    print(f"  [debug] raw response: {raw_text[:300]}")

    try:
        data = extract_json(raw_text)
        return ImageTags(**data)
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        if retry_with_error is None:
            print(f"  First attempt failed: {e}")
            print("  Retrying once...")
            return tag_image(image_path, retry_with_error=str(e))
        else:
            print(f"  Repair attempt also failed: {e}")
            
            return None

if __name__ == "__main__":
    test_images = [
        "images/wolf_1.jpg",
        "images/red-fox_1.jpg",
        "images/dog_1.jpg",
        "images/deer_1.jpg",
        "images/bear_1.jpg",
    ]
    for img in test_images:
        print(f"\n=== {img} ===")
        result = tag_image(img)
        if result:
            print(result.model_dump_json(indent=2))
        else:
            print("Failed to get valid tags — would be flagged in the real pipeline")