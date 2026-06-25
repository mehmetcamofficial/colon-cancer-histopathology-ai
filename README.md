# Colon Cancer Histopathology AI — Research Demo

A research-oriented AI demo for colon histopathology image classification using **EfficientNet-B0**, **Grad-CAM explainability**, **OpenRouter o4-mini safety reporting**, and a **human-in-the-loop feedback workflow**.

Live demo: https://colon-cancer-histopathology-ai.streamlit.app/
Project owner: Mehmet Cam
Website: https://oncoconnectai.com.tr/
Repository: https://github.com/mehmetcamofficial/colon-cancer-histopathology-ai

---

## Important Medical Disclaimer

This application is for **research and educational demonstration only**.

It is **not** intended for:

* clinical diagnosis,
* treatment planning,
* medical decision-making,
* pathology replacement,
* regulatory or hospital use.

All outputs must be interpreted as experimental model behavior. Final medical evaluation must always be performed by qualified pathology specialists.

---

## Project Overview

This project started as a colon histopathology image classification demo and has been expanded into a more complete research workflow.

The system can:

* classify colon histopathology image patches as **cancer-like** or **normal/benign-like**,
* generate Grad-CAM visual explanations,
* produce a cautious AI interpretation report using OpenRouter o4-mini,
* collect human feedback,
* support manual/expert review,
* build reviewed datasets with patient/slide-aware splitting,
* train reviewed models,
* evaluate models on external test datasets,
* analyze large histology-like images through patch extraction,
* generate slide-level patch heatmaps,
* calculate research-only slide-level risk scores,
* manage active model promotion through a lightweight model registry,
* compare models through a dashboard.

---

## Current Features

### 1. Image Classification

The app supports single or multiple histopathology image uploads.

For each image, the model produces:

* predicted class,
* confidence score,
* cancer-like probability,
* normal-like probability.

Current classes:

```text
cancer_like
normal_like
```

---

### 2. Grad-CAM Explainability

The app generates:

* original image,
* Grad-CAM heatmap,
* approximate attention zone.

Important: Grad-CAM does **not** show a confirmed tumor boundary. It only shows where the model focused when making a prediction.

---

### 3. AI Interpretation Report

The app can generate a cautious explanation using OpenRouter o4-mini.

The report includes:

* brief summary,
* model output interpretation,
* safety and limitations,
* suggested next step.

The report is non-diagnostic and explicitly warns that the tool is a research demo.

---

### 4. Human-in-the-loop Feedback Collection

After prediction, users can submit feedback:

```text
Correct
Incorrect
Unsure
```

Optional labels:

```text
cancer_like
normal_like
not_histopathology
not_sure
```

Feedback is saved as candidate data for later review. The model does **not** learn automatically from user uploads.

---

### 5. Admin Review Panel

A separate admin review page supports manual or expert review.

Review statuses:

```text
pending_review
approved_cancer_like
approved_normal_like
rejected
needs_more_info
```

This enables a safer research workflow:

```text
user upload
→ model prediction
→ user feedback
→ candidate sample
→ manual/expert review
→ approved retraining candidate
```

---

### 6. Slide/patient-aware Dataset Builder

Approved samples can be exported into:

```text
reviewed_dataset/
├── train/
│   ├── cancer_like/
│   └── normal_like/
├── val/
│   ├── cancer_like/
│   └── normal_like/
└── test/
    ├── cancer_like/
    └── normal_like/
```

The split logic uses:

```text
patient_id
slide_id
image_id fallback
```

The goal is to reduce data leakage by keeping samples from the same patient or slide within the same split.

---

### 7. Reviewed Model Training Pipeline

The project includes a training script for reviewed datasets:

```bash
python train_reviewed_model.py
```

The script performs:

* EfficientNet-B0 fine-tuning,
* validation monitoring,
* cancer recall tracking,
* best model saving,
* test evaluation,
* confusion matrix generation,
* training curve generation,
* metrics JSON export.

---

### 8. External Evaluation Pipeline

Independent evaluation can be run on an external dataset:

```bash
python evaluate_external_model.py \
  --model models/colon_cancer_efficientnet_b0.pth \
  --data external_test_dataset
```

Expected external dataset format:

```text
external_test_dataset/
├── cancer_like/
└── normal_like/
```

The evaluation outputs:

```text
metrics/external/
├── metrics_external_YYYYMMDD_HHMMSS.json
├── confusion_matrix_external_YYYYMMDD_HHMMSS.png
└── predictions_external_YYYYMMDD_HHMMSS.csv
```

---

### 9. Large Image / WSI-like Patch Pipeline

The project supports lightweight large-image analysis.

Workflow:

```text
large histology-like image
→ patch extraction
→ background filtering
→ patch-level prediction
→ slide-level heatmap
→ slide-level risk score
```

This is not a full clinical WSI pipeline and does not yet support production-level `.svs` processing.

---

### 10. Model Registry and Promotion

The project includes a simple model registry.

A reviewed model can be promoted as the active model:

```bash
python promote_model.py \
  --model models/reviewed/colon_cancer_reviewed_vYYYYMMDD_HHMMSS.pth \
  --name reviewed-v1 \
  --metrics metrics/reviewed/metrics_vYYYYMMDD_HHMMSS.json \
  --reason "Promoted after reviewed dataset training and external evaluation."
```

The app then loads:

```text
models/active_model.pth
```

if available.

---

### 11. Model Comparison Dashboard

The Streamlit app includes a model comparison page showing:

* active model,
* registered models,
* reviewed training metrics,
* external evaluation metrics,
* best candidate by cancer recall,
* related charts,
* registry JSON.

---

## Dataset

The initial model was trained using the LC25000 colon histopathology subset.

Classes used:

```text
colon_aca → cancer_like
colon_n   → normal_like
```

Important limitation: LC25000 contains augmented image patches. High performance on this dataset should not be interpreted as real-world clinical reliability.

---

## Baseline Results

Initial baseline:

```text
Model: ResNet18
Test Accuracy: 97.60%
Cancer Recall: 95.00%
False Negatives: 35
```

Improved model:

```text
Model: EfficientNet-B0
Test Accuracy: 100.00%
Cancer Recall: 100.00%
False Negatives: 0
```

Important: The 100% result is likely influenced by the structure of the dataset and should not be treated as clinical-level validation.

---

## Scientific Motivation

A key lesson from medical imaging research is that model performance can be overestimated when data splitting is not patient-aware or slide-aware.

This project therefore adds:

* human-in-the-loop review,
* patient/slide metadata,
* reviewed dataset export,
* external evaluation,
* model comparison,
* cautious reporting.

The goal is not to claim clinical readiness, but to demonstrate a more responsible research workflow.

---

## Project Structure

```text
colon-cancer-histopathology-ai/
├── app.py
├── model.py
├── feedback_store.py
├── build_reviewed_dataset.py
├── train_reviewed_model.py
├── evaluate_external_model.py
├── large_image_pipeline.py
├── model_registry.py
├── promote_model.py
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── pages/
│   ├── 1_Admin_Review.py
│   └── 2_Model_Comparison.py
├── assets/
│   ├── efficientnet_confusion_matrix.png
│   ├── efficientnet_accuracy.png
│   ├── efficientnet_cancer_recall.png
│   └── efficientnet_loss.png
├── models/
│   └── colon_cancer_efficientnet_b0.pth
└── model_registry/
    └── registry.json
```

Generated/local folders are ignored:

```text
feedback/
feedback_images/
reviewed_dataset/
external_test_dataset/
models/reviewed/
models/active_model.pth
metrics/reviewed/
metrics/external/
large_image_outputs/
```

---

## Installation

```bash
git clone https://github.com/mehmetcamofficial/colon-cancer-histopathology-ai.git
cd colon-cancer-histopathology-ai

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run the Streamlit App

```bash
streamlit run app.py
```

---

## Streamlit Secrets

For OpenRouter report generation:

```toml
OPENROUTER_API_KEY = "sk-or-..."
ADMIN_PASSWORD = "your-admin-password"
```

Local file:

```text
.streamlit/secrets.toml
```

This file must not be committed.

---

## Human-in-the-loop Workflow

```text
1. User uploads image
2. Model predicts cancer-like / normal-like
3. User submits feedback
4. Candidate sample is saved
5. Admin/manual review approves or rejects sample
6. Approved samples are exported into reviewed_dataset
7. Reviewed model is trained
8. External evaluation is performed
9. Safer model can be promoted
10. Model comparison dashboard tracks results
```

---

## Large Image Workflow

```text
large histology-like image
→ 224x224 patch extraction
→ tissue/background filtering
→ patch-level prediction
→ heatmap overlay
→ slide-level research score
```

The slide-level risk score is a research aggregation only. It is not a clinical score.

---

## Current Limitations

* Not clinically validated.
* Not approved for diagnosis.
* Initial model trained on an augmented public dataset.
* External validation data is still required.
* Patient/slide metadata depends on user/admin input.
* Grad-CAM is not a segmentation mask.
* Large image mode is WSI-like, not full production WSI support.
* `.svs` support is not production-ready yet.
* User feedback is not expert ground truth unless reviewed.
* Model promotion should only happen after external evaluation.

---

## Roadmap

Planned next steps:

* Add stronger model card reporting.
* Add external dataset examples.
* Add model comparison summary export.
* Add proper `.svs` support using OpenSlide.
* Add tissue detection improvements.
* Add patch-level uncertainty.
* Add calibration metrics.
* Add ROC-AUC and PR-AUC.
* Add model drift monitoring.
* Add dataset versioning.
* Add automated evaluation reports.
* Add privacy and consent notes for real-world data.
* Add documentation for pathologist review workflow.

---

## Responsible AI Position

This project is designed as a responsible medical AI research demo.

It intentionally avoids claims such as:

```text
diagnoses cancer
detects tumor boundaries
replaces pathology review
clinically validated system
```

Instead, it uses safer language:

```text
cancer-like tissue pattern
normal/benign-like tissue pattern
model attention region
research-only risk score
candidate retraining data
human-in-the-loop review
```

---

## License

See `LICENSE`.

---

## Author

Mehmet Cam
OncoConnect AI
https://oncoconnectai.com.tr/
