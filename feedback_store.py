import csv
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image


FEEDBACK_DIR = Path("feedback")
FEEDBACK_IMAGES_DIR = Path("feedback_images")
FEEDBACK_CSV_PATH = FEEDBACK_DIR / "feedback.csv"


CSV_COLUMNS = [
    "timestamp",
    "image_id",
    "original_filename",
    "saved_image_path",
    "model_prediction",
    "confidence",
    "cancer_probability",
    "normal_probability",
    "user_feedback",
    "user_label",
    "review_status",
    "notes",
]


def ensure_feedback_storage():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not FEEDBACK_CSV_PATH.exists():
        with FEEDBACK_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def save_feedback_sample(
    image: Image.Image,
    original_filename: str,
    result: dict,
    user_feedback: str,
    user_label: str,
    notes: str = "",
):
    """
    Saves uploaded image + model prediction + user feedback as candidate data.

    Important:
    This does NOT retrain the model automatically.
    It only stores candidate samples for later human review and controlled retraining.
    """

    ensure_feedback_storage()

    image_id = str(uuid.uuid4())
    safe_name = original_filename.replace("/", "_").replace("\\", "_")
    saved_image_path = FEEDBACK_IMAGES_DIR / f"{image_id}_{safe_name}.png"

    image.convert("RGB").save(saved_image_path)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_id": image_id,
        "original_filename": original_filename,
        "saved_image_path": str(saved_image_path),
        "model_prediction": result.get("label", ""),
        "confidence": round(float(result.get("confidence", 0)), 6),
        "cancer_probability": round(float(result.get("cancer_probability", 0)), 6),
        "normal_probability": round(float(result.get("normal_probability", 0)), 6),
        "user_feedback": user_feedback,
        "user_label": user_label,
        "review_status": "pending_review",
        "notes": notes,
    }

    with FEEDBACK_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

    return row


def feedback_csv_exists():
    return FEEDBACK_CSV_PATH.exists()


def read_feedback_csv_bytes():
    if not FEEDBACK_CSV_PATH.exists():
        return None

    return FEEDBACK_CSV_PATH.read_bytes()