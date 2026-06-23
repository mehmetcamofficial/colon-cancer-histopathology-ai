import streamlit as st
from PIL import Image
from model import load_model, predict


st.set_page_config(
    page_title="Colon Cancer Histopathology AI Demo",
    page_icon="🧬",
    layout="wide"
)


@st.cache_resource
def cached_model():
    return load_model()


model, device = cached_model()


st.title("🧬 Colon Cancer Histopathology AI Demo")
st.caption("EfficientNet-B0 model for colon adenocarcinoma vs benign colon tissue classification.")

st.warning(
    "Medical disclaimer: This application is for research and educational demonstration only. "
    "It is not intended for clinical diagnosis or medical decision-making. "
    "Final evaluation must always be performed by qualified pathology specialists."
)

left, right = st.columns([1, 1])

with left:
    st.subheader("Upload a histopathology image")
    uploaded_file = st.file_uploader(
        "Upload JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        result = predict(image, model, device)

        st.subheader("Prediction")

        if result["label"] == "cancer":
            st.error(f"Prediction: CANCER")
        else:
            st.success(f"Prediction: NORMAL")

        st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
        st.progress(result["confidence"])

        st.write("Class probabilities:")
        st.write({
            "cancer": f"{result['cancer_probability'] * 100:.2f}%",
            "normal": f"{result['normal_probability'] * 100:.2f}%"
        })

with right:
    st.subheader("Model results")

    st.markdown(
        """
        **Dataset:** LC25000 colon histopathology subset  
        **Classes:** colon adenocarcinoma vs benign colon tissue  
        **Model:** EfficientNet-B0  
        **Test Accuracy:** 100.00%  
        **Cancer Recall:** 100.00%  
        **False Negatives:** 0 on the test split  
        """
    )

    st.image(
        "assets/efficientnet_confusion_matrix.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

st.divider()

st.subheader("Training Curves")

c1, c2, c3 = st.columns(3)

with c1:
    st.image("assets/efficientnet_accuracy.png", caption="Accuracy", use_container_width=True)

with c2:
    st.image("assets/efficientnet_cancer_recall.png", caption="Cancer Recall", use_container_width=True)

with c3:
    st.image("assets/efficientnet_loss.png", caption="Loss", use_container_width=True)

st.divider()

st.markdown(
    """
    ### Important limitation

    The test results are strong, but they should not be interpreted as clinical performance.
    The LC25000 dataset contains augmented images, and real-world pathology data may differ
    in staining, scanner quality, tissue preparation, and patient distribution.
    """
)