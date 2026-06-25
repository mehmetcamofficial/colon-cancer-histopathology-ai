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
    "patient_id",
    "slide_id",
    "review_status",
    "reviewed_label",
    "reviewer",
    "review_notes",
    "reviewed_at",
    "notes",
]


def ensure_feedback_storage():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not FEEDBACK_CSV_PATH.exists():
        with FEEDBACK_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        return

    migrate_feedback_csv_if_needed()


def migrate_feedback_csv_if_needed():
    """
    Adds missing columns if feedback.csv was created by an older version.
    """
    if not FEEDBACK_CSV_PATH.exists():
        return

    rows = read_feedback_rows()

    needs_migration = False
    for row in rows:
        for col in CSV_COLUMNS:
            if col not in row:
                row[col] = ""
                needs_migration = True

    if needs_migration:
        write_feedback_rows(rows)


def read_feedback_rows():
    ensure_feedback_dir_only()

    if not FEEDBACK_CSV_PATH.exists():
        return []

    with FEEDBACK_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    normalized_rows = []
    for row in rows:
        normalized = {col: row.get(col, "") for col in CSV_COLUMNS}
        normalized_rows.append(normalized)

    return normalized_rows


def write_feedback_rows(rows):
    ensure_feedback_dir_only()

    with FEEDBACK_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row in rows:
            normalized = {col: row.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(normalized)


def ensure_feedback_dir_only():
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def save_feedback_sample(
    image: Image.Image,
    original_filename: str,
    result: dict,
    user_feedback: str,
    user_label: str,
    notes: str = "",
    patient_id: str = "",
    slide_id: str = "",
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
        "patient_id": patient_id,
        "slide_id": slide_id,
        "review_status": "pending_review",
        "reviewed_label": "",
        "reviewer": "",
        "review_notes": "",
        "reviewed_at": "",
        "notes": notes,
    }

    rows = read_feedback_rows()
    rows.append(row)
    write_feedback_rows(rows)

    return row


def update_feedback_review(
    image_id: str,
    review_status: str,
    reviewed_label: str,
    reviewer: str = "",
    review_notes: str = "",
    patient_id: str = "",
    slide_id: str = "",
):
    """
    Updates a candidate sample after manual or expert review.
    """

    ensure_feedback_storage()
    rows = read_feedback_rows()

    updated_row = None

    for row in rows:
        if row.get("image_id") == image_id:
            row["review_status"] = review_status
            row["reviewed_label"] = reviewed_label
            row["reviewer"] = reviewer
            row["review_notes"] = review_notes
            row["reviewed_at"] = datetime.now().isoformat(timespec="seconds")

            if patient_id:
                row["patient_id"] = patient_id

            if slide_id:
                row["slide_id"] = slide_id

            updated_row = row
            break

    write_feedback_rows(rows)

    return updated_row


def get_feedback_summary():
    ensure_feedback_storage()
    rows = read_feedback_rows()

    summary = {
        "total": len(rows),
        "pending_review": 0,
        "approved_cancer_like": 0,
        "approved_normal_like": 0,
        "rejected": 0,
        "needs_more_info": 0,
    }

    for row in rows:
        status = row.get("review_status", "pending_review")

        if status in summary:
            summary[status] += 1
        else:
            summary[status] = summary.get(status, 0) + 1

    return summary


def get_rows_by_status(status=None):
    ensure_feedback_storage()
    rows = read_feedback_rows()

    if status is None or status == "all":
        return rows

    return [row for row in rows if row.get("review_status") == status]


def feedback_csv_exists():
    return FEEDBACK_CSV_PATH.exists()


def read_feedback_csv_bytes():
    if not FEEDBACK_CSV_PATH.exists():
        return None

    return FEEDBACK_CSV_PATH.read_bytes()


def export_approved_dataset_rows():
    """
    Returns only reviewed/approved samples.
    These are the only candidates that should be used for controlled retraining.
    """

    rows = read_feedback_rows()

    approved_statuses = {
        "approved_cancer_like",
        "approved_normal_like",
    }

    return [row for row in rows if row.get("review_status") in approved_statuses]