# Colon Cancer Histopathology AI Demo

An EfficientNet-B0 based image classification demo for distinguishing colon adenocarcinoma tissue from benign colon tissue using the LC25000 histopathology dataset.

## Model

- Architecture: EfficientNet-B0
- Task: Binary image classification
- Classes:
  - cancer
  - normal
- Input size: 224x224
- Framework: PyTorch
- Demo: Streamlit

## Results

### Baseline ResNet18

- Test Accuracy: 97.60%
- Cancer Recall: 95.00%
- False Negatives: 35

### Improved EfficientNet-B0

- Test Accuracy: 100.00%
- Cancer Recall: 100.00%
- False Negatives: 0

## Important Medical Disclaimer

This project is for research and educational demonstration only. It is not intended for clinical diagnosis or medical decision-making. The dataset is augmented and may not represent real-world clinical distribution. Final diagnosis must always be made by qualified pathology specialists.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py

## Localde test et

Terminalde:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

http://localhost:8501