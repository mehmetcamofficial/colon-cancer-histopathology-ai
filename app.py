import streamlit as st
from PIL import Image
from model import load_model, predict


st.set_page_config(
    page_title="OncoConnect AI | Colon Pathology Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def cached_model():
    return load_model()


model, device = cached_model()


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 28%),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 45%, #ffffff 100%);
    }

    .main .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    .hero-card {
        padding: 2.4rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0f766e 100%);
        color: white;
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.25);
        margin-bottom: 1.5rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.42rem 0.78rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 850;
        margin: 0 0 1rem 0;
        letter-spacing: -0.05em;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        line-height: 1.7;
        color: rgba(255, 255, 255, 0.86);
        max-width: 820px;
        margin-bottom: 1.25rem;
    }

    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.4rem;
    }

    .meta-pill {
        padding: 0.62rem 0.9rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.20);
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.92);
    }

    .glass-card {
        padding: 1.4rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.55rem;
        letter-spacing: -0.025em;
    }

    .muted {
        color: #64748b;
        line-height: 1.65;
        font-size: 0.98rem;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 850;
        letter-spacing: -0.04em;
    }

    .prediction-card-cancer {
        padding: 1.35rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(239, 68, 68, 0.25);
        margin-top: 1rem;
    }

    .prediction-card-normal {
        padding: 1.35rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #047857 0%, #10b981 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(16, 185, 129, 0.25);
        margin-top: 1rem;
    }

    .prediction-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.88;
        font-weight: 800;
    }

    .prediction-main {
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-top: 0.15rem;
    }

    .prediction-confidence {
        margin-top: 0.3rem;
        font-size: 1.05rem;
        opacity: 0.93;
    }

    .disclaimer {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        line-height: 1.6;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    div[data-testid="stFileUploader"] {
        padding: 0.75rem;
        border-radius: 20px;
        background: #f8fafc;
        border: 1px dashed #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 850;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.15rem;
        }
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">🧬 OncoConnect AI Research Demo</div>
        <div class="hero-title">Colon Cancer Histopathology AI</div>
        <div class="hero-subtitle">
            An EfficientNet-B0 based image classification demo designed to distinguish
            colon adenocarcinoma tissue from benign colon tissue using histopathology images.
            Built as a research and educational AI prototype.
        </div>
        <div class="hero-meta">
            <div class="meta-pill">Model: EfficientNet-B0</div>
            <div class="meta-pill">Dataset: LC25000 colon subset</div>
            <div class="meta-pill">Task: Binary image classification</div>
            <div class="meta-pill">Framework: PyTorch + Streamlit</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="disclaimer">
        <strong>Medical disclaimer:</strong>
        This application is for research and educational demonstration only.
        It is not intended for clinical diagnosis, treatment planning, or medical decision-making.
        Final evaluation must always be performed by qualified pathology specialists.
    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([1.02, 0.98], gap="large")


with left:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Upload a histopathology image</div>
            <div class="muted">
                Upload a JPG, JPEG, or PNG tissue image. The model will classify it as
                <strong>cancer</strong> or <strong>normal</strong> and report class probabilities.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded histopathology image", use_container_width=True)

        result = predict(image, model, device)

        if result["label"] == "cancer":
            st.markdown(
                f"""
                <div class="prediction-card-cancer">
                    <div class="prediction-label">Prediction result</div>
                    <div class="prediction-main">CANCER / Adenocarcinoma-like tissue</div>
                    <div class="prediction-confidence">Confidence: {result["confidence"] * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-card-normal">
                    <div class="prediction-label">Prediction result</div>
                    <div class="prediction-main">NORMAL / Benign-like tissue</div>
                    <div class="prediction-confidence">Confidence: {result["confidence"] * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Cancer probability", f"{result['cancer_probability'] * 100:.2f}%")
        with p2:
            st.metric("Normal probability", f"{result['normal_probability'] * 100:.2f}%")

        st.progress(result["confidence"])
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">How to test</div>
                <div class="muted">
                    Use a colon histopathology sample image. For a reliable demo, use images
                    similar to the LC25000 colon tissue format. Non-histology photos are outside
                    the model's intended distribution.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with right:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Model performance</div>
            <div class="muted">
                The improved EfficientNet-B0 model was selected because it reduced false negatives
                on the test split compared with the ResNet18 baseline.
            </div>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Test Accuracy</div>
                    <div class="metric-value">100%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cancer Recall</div>
                    <div class="metric-value">100%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">False Negatives</div>
                    <div class="metric-value">0</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image(
        "assets/efficientnet_confusion_matrix.png",
        caption="EfficientNet-B0 Confusion Matrix",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")

st.markdown(
    """
    <div class="glass-card">
        <div class="section-title">Training curves</div>
        <div class="muted">
            Accuracy, cancer recall, and loss curves from the EfficientNet-B0 training run.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image("assets/efficientnet_accuracy.png", caption="Accuracy", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image("assets/efficientnet_cancer_recall.png", caption="Cancer Recall", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image("assets/efficientnet_loss.png", caption="Loss", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="glass-card">
        <div class="section-title">Important limitation</div>
        <div class="muted">
            The LC25000 dataset contains augmented images. Therefore, the reported test results
            should not be interpreted as real-world clinical performance. Real clinical deployment
            would require external validation, expert review, regulatory evaluation, and testing
            across different scanners, staining protocols, and patient populations.
        </div>
    </div>

    <div class="footer-note">
        Built by Mehmet Cam · OncoConnect AI · Research prototype
    </div>
    """,
    unsafe_allow_html=True,
)