import csv
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from feedback_store import (
    get_feedback_summary,
    get_rows_by_status,
    update_feedback_review,
    read_feedback_csv_bytes,
    export_approved_dataset_rows,
)


st.set_page_config(
    page_title="OncoConnect AI | Admin Review",
    page_icon="🧬",
    layout="wide",
)


def require_admin_password():
    """
    Simple demo-level admin protection.
    For production, use real authentication.
    """

    expected_password = st.secrets.get("ADMIN_PASSWORD", None)

    if not expected_password:
        st.warning(
            "ADMIN_PASSWORD is not set. Add it to Streamlit Secrets to protect this page."
        )
        return True

    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    if st.session_state["admin_authenticated"]:
        return True

    st.title("🔐 Admin Review Login")
    password = st.text_input("Admin password", type="password")

    if st.button("Login", type="primary"):
        if password == expected_password:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Wrong password.")

    return False


def rows_to_csv_bytes(rows):
    if not rows:
        return b""

    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue().encode("utf-8")


if not require_admin_password():
    st.stop()


st.title("🧬 OncoConnect AI — Human-in-the-loop Review Panel")

st.markdown(
    """
This page is for reviewing candidate samples collected from user feedback.

The model **does not learn automatically** from uploaded images.  
Only manually reviewed and approved samples should be used later for controlled retraining.
"""
)


summary = get_feedback_summary()

s1, s2, s3, s4, s5 = st.columns(5)

with s1:
    st.metric("Total", summary.get("total", 0))

with s2:
    st.metric("Pending", summary.get("pending_review", 0))

with s3:
    st.metric("Approved cancer-like", summary.get("approved_cancer_like", 0))

with s4:
    st.metric("Approved normal-like", summary.get("approved_normal_like", 0))

with s5:
    st.metric("Rejected", summary.get("rejected", 0))


st.divider()


filter_status = st.selectbox(
    "Filter by review status",
    [
        "pending_review",
        "approved_cancer_like",
        "approved_normal_like",
        "rejected",
        "needs_more_info",
        "all",
    ],
)


rows = get_rows_by_status(filter_status)

if not rows:
    st.info("No feedback samples found for this status.")
else:
    st.subheader("Candidate samples")

    df = pd.DataFrame(rows)
    visible_cols = [
        "timestamp",
        "image_id",
        "original_filename",
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
    ]

    available_cols = [col for col in visible_cols if col in df.columns]
    st.dataframe(df[available_cols], use_container_width=True)

    st.divider()

    for idx, row in enumerate(rows):
        image_id = row.get("image_id", "")
        title = f"{idx + 1}. {row.get('original_filename', 'unknown')} — {row.get('review_status', '')}"

        with st.expander(title, expanded=(idx == 0)):
            left, right = st.columns([0.9, 1.1], gap="large")

            with left:
                image_path = row.get("saved_image_path", "")

                if image_path and Path(image_path).exists():
                    image = Image.open(image_path).convert("RGB")
                    st.image(image, caption=row.get("original_filename", ""), use_container_width=True)
                else:
                    st.warning("Image file not found. This can happen if storage was reset or not persisted.")

            with right:
                st.markdown("### Model output")

                st.write(f"**Image ID:** `{image_id}`")
                st.write(f"**Model prediction:** `{row.get('model_prediction', '')}`")
                st.write(f"**Confidence:** {float(row.get('confidence') or 0) * 100:.2f}%")
                st.write(f"**Cancer-like probability:** {float(row.get('cancer_probability') or 0) * 100:.2f}%")
                st.write(f"**Normal-like probability:** {float(row.get('normal_probability') or 0) * 100:.2f}%")

                st.markdown("### User feedback")
                st.write(f"**User feedback:** `{row.get('user_feedback', '')}`")
                st.write(f"**User label:** `{row.get('user_label', '')}`")
                st.write(f"**User notes:** {row.get('notes', '')}")

                st.markdown("### Manual / expert review")

                patient_id = st.text_input(
                    "Patient ID / case ID",
                    value=row.get("patient_id", ""),
                    key=f"patient_{image_id}",
                    help="Use anonymized IDs only. Do not enter personal health information.",
                )

                slide_id = st.text_input(
                    "Slide ID",
                    value=row.get("slide_id", ""),
                    key=f"slide_{image_id}",
                    help="This will later help with slide-aware splitting.",
                )

                review_status = st.selectbox(
                    "Review decision",
                    [
                        "pending_review",
                        "approved_cancer_like",
                        "approved_normal_like",
                        "rejected",
                        "needs_more_info",
                    ],
                    index=[
                        "pending_review",
                        "approved_cancer_like",
                        "approved_normal_like",
                        "rejected",
                        "needs_more_info",
                    ].index(row.get("review_status", "pending_review"))
                    if row.get("review_status", "pending_review")
                    in [
                        "pending_review",
                        "approved_cancer_like",
                        "approved_normal_like",
                        "rejected",
                        "needs_more_info",
                    ]
                    else 0,
                    key=f"status_{image_id}",
                )

                reviewed_label = st.selectbox(
                    "Reviewed label",
                    [
                        "",
                        "cancer_like",
                        "normal_like",
                        "not_histopathology",
                        "uncertain",
                    ],
                    index=[
                        "",
                        "cancer_like",
                        "normal_like",
                        "not_histopathology",
                        "uncertain",
                    ].index(row.get("reviewed_label", ""))
                    if row.get("reviewed_label", "")
                    in [
                        "",
                        "cancer_like",
                        "normal_like",
                        "not_histopathology",
                        "uncertain",
                    ]
                    else 0,
                    key=f"label_{image_id}",
                )

                reviewer = st.text_input(
                    "Reviewer",
                    value=row.get("reviewer", ""),
                    key=f"reviewer_{image_id}",
                )

                review_notes = st.text_area(
                    "Review notes",
                    value=row.get("review_notes", ""),
                    key=f"review_notes_{image_id}",
                )

                if st.button(
                    "Save review decision",
                    type="primary",
                    use_container_width=True,
                    key=f"save_{image_id}",
                ):
                    updated = update_feedback_review(
                        image_id=image_id,
                        review_status=review_status,
                        reviewed_label=reviewed_label,
                        reviewer=reviewer,
                        review_notes=review_notes,
                        patient_id=patient_id,
                        slide_id=slide_id,
                    )

                    if updated:
                        st.success("Review decision saved.")
                        st.rerun()
                    else:
                        st.error("Could not update this row.")


st.divider()

st.subheader("Exports")

feedback_bytes = read_feedback_csv_bytes()

if feedback_bytes:
    st.download_button(
        label="Download full feedback CSV",
        data=feedback_bytes,
        file_name="oncoconnect_feedback_full.csv",
        mime="text/csv",
        use_container_width=True,
    )

approved_rows = export_approved_dataset_rows()

if approved_rows:
    approved_csv = rows_to_csv_bytes(approved_rows)

    st.download_button(
        label="Download approved retraining candidates CSV",
        data=approved_csv,
        file_name="oncoconnect_approved_retraining_candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("No approved retraining candidates yet.")