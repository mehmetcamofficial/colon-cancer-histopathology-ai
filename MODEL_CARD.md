# Model Card — Colon Cancer Histopathology AI

## Model Name

Colon Cancer Histopathology AI — EfficientNet-B0 Research Demo

---

## Model Type

Image classification model for colon histopathology image patches.

Architecture:

```text
EfficientNet-B0
ImageNet pretrained backbone
Custom binary classification head
```

Framework:

```text
PyTorch
Torchvision
Streamlit
```

---

## Intended Use

This model is intended for:

* educational demonstration,
* research prototyping,
* AI-assisted histopathology workflow exploration,
* portfolio and hackathon demonstration,
* studying human-in-the-loop medical AI workflows.

The model is not intended for:

* clinical diagnosis,
* patient treatment,
* automated pathology decision-making,
* hospital deployment,
* regulatory use,
* replacing a pathologist.

---

## Output Classes

```text
cancer_like
normal_like
```

These labels should be interpreted as model categories, not clinical truth.

Safer language:

```text
cancer-like tissue pattern
normal/benign-like tissue pattern
```

Avoid:

```text
patient has cancer
tumor detected
clinical diagnosis
```

---

## Dataset Used

Initial training used the LC25000 colon histopathology subset.

Class mapping:

```text
colon_aca → cancer_like
colon_n   → normal_like
```

Dataset limitation:

LC25000 contains augmented image patches. Performance on this dataset may overestimate real-world model reliability.

---

## Training Summary

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

Important caveat:

The 100% result should not be interpreted as real clinical performance. The dataset structure and augmentation may make the task easier than real-world histopathology deployment.

---

## Evaluation Metrics

The project prioritizes:

* cancer recall,
* false negatives,
* confusion matrix,
* accuracy,
* precision,
* F1-score,
* external evaluation results.

For medical AI research, cancer recall and false negatives are especially important because missing cancer-like regions may be more harmful than false positives.

---

## Explainability

The app uses Grad-CAM to visualize model attention.

Grad-CAM output means:

```text
areas that contributed to the model prediction
```

Grad-CAM output does not mean:

```text
confirmed tumor boundary
pathologist-verified malignancy
segmentation mask
clinical annotation
```

---

## Human-in-the-loop Feedback

The system supports feedback collection after prediction.

User feedback options:

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

Important:

The model does not learn automatically from uploaded images. Feedback is stored as candidate data and requires manual or expert review before any retraining.

---

## Manual / Expert Review

The admin review panel supports:

```text
pending_review
approved_cancer_like
approved_normal_like
rejected
needs_more_info
```

Approved samples can later be exported for controlled retraining.

---

## Patient / Slide-aware Splitting

The reviewed dataset builder supports grouping by:

```text
patient_id
slide_id
image_id fallback
```

This reduces the risk of leakage between train, validation, and test splits.

Scientific motivation:

If patches from the same patient or slide appear in both training and testing sets, the model may appear stronger than it really is.

---

## External Evaluation

The project includes an external evaluation script:

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

External evaluation is required before making any stronger claims about generalization.

---

## Large Image / WSI-like Analysis

The app supports large histology-like image analysis through patch extraction.

Workflow:

```text
large image
→ patch extraction
→ background filtering
→ patch-level prediction
→ heatmap
→ slide-level research score
```

This is not yet a full clinical WSI pipeline.

Current limitations:

* no production `.svs` support,
* no validated tissue segmentation,
* no pathologist-confirmed region-level labels,
* no clinical slide-level validation.

---

## Model Registry

The project includes a lightweight registry for active model selection.

A model can be promoted using:

```bash
python promote_model.py \
  --model models/reviewed/colon_cancer_reviewed_vYYYYMMDD_HHMMSS.pth \
  --name reviewed-v1 \
  --metrics metrics/reviewed/metrics_vYYYYMMDD_HHMMSS.json \
  --reason "Promoted after reviewed dataset training and external evaluation."
```

The app loads:

```text
models/active_model.pth
```

if available.

---

## Known Risks

### 1. Dataset bias

The model may learn staining, scanner, augmentation, or dataset-specific artifacts rather than robust histopathology features.

### 2. Data leakage

Patch-level random splitting can overestimate performance if patches from the same slide or patient appear across train/test splits.

### 3. Overconfidence

High softmax confidence does not guarantee clinical correctness.

### 4. Explainability limitations

Grad-CAM can be visually helpful but does not provide verified pathology reasoning.

### 5. Feedback noise

User feedback may be incorrect unless reviewed by qualified experts.

### 6. Domain shift

Performance may degrade on images from different hospitals, scanners, staining protocols, or preparation workflows.

---

## Ethical and Safety Considerations

This project must not be presented as a clinical diagnostic tool.

Safe framing:

```text
research demo
AI-assisted exploration
model attention visualization
candidate feedback workflow
human-in-the-loop review
```

Unsafe framing:

```text
AI diagnoses cancer
automatic cancer detection for patients
clinical decision system
doctor replacement
```

---

## Recommended Validation Before Any Clinical Claim

Before any clinical or near-clinical claim, the system would require:

* external validation,
* multi-center data,
* pathologist-reviewed labels,
* patient/slide-aware splitting,
* calibration analysis,
* failure mode analysis,
* subgroup analysis,
* robustness testing,
* regulatory review,
* privacy and consent process,
* clinical workflow evaluation.

---

## Current Model Status

```text
Status: Research prototype
Clinical readiness: No
External validation: Required
Pathologist validation: Required
Regulatory approval: None
```

---

## Appropriate Citation / Description

Suggested description:

> This project is a research prototype for colon histopathology image classification using EfficientNet-B0, Grad-CAM explainability, OpenRouter-based safety reporting, and human-in-the-loop feedback collection. It is not a clinical diagnostic system.

---

## Versioning

Model versions should be tracked through:

```text
models/
model_registry/
metrics/
```

Each promoted model should have:

* model checkpoint,
* metrics JSON,
* confusion matrix,
* training/evaluation notes,
* promotion reason.

---

## Final Note

This model card is part of the responsible AI documentation for the project. It is designed to make limitations, risks, and intended use clear.
