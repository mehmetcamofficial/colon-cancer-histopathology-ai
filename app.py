import textwrap
import streamlit as st
from PIL import Image
from model import load_model, predict


GITHUB_URL = "https://github.com/mehmetcamofficial/colon-cancer-histopathology-ai"
ONCOCONNECT_URL = "https://oncoconnectai.com.tr/"


st.set_page_config(
    page_title="OncoConnect AI | Colon Pathology Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def html(content: str):
    cleaned = "\n".join(
        line.strip()
        for line in textwrap.dedent(content).strip().splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


@st.cache_resource
def cached_model():
    return load_model()


model, device = cached_model()


with st.sidebar:
    st.markdown("## 🧬 OncoConnect AI")

    lang = st.radio(
        "Language / Dil",
        ["English", "Türkçe"],
        horizontal=True,
    )

    theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        horizontal=True,
    )

    mode = st.selectbox(
        "Mode",
        ["Research Mode", "Demo Mode"],
    )

    st.divider()
    st.link_button("🌐 OncoConnect Website", ONCOCONNECT_URL, use_container_width=True)
    st.link_button("💻 GitHub Repository", GITHUB_URL, use_container_width=True)


is_tr = lang == "Türkçe"
is_dark = theme == "Dark"


if is_tr:
    T = {
        "badge": "OncoConnect AI Araştırma Prototipi",
        "title": "Kolon Kanseri Histopatoloji AI",
        "subtitle": "EfficientNet-B0 tabanlı bu demo, histopatoloji görüntülerinde kolon adenokarsinom dokusu ile benign/normal kolon dokusunu ayırt etmek için geliştirilmiş bir yapay zekâ araştırma prototipidir.",
        "upload_title": "Histopatoloji görüntüsü yükle",
        "upload_desc": "JPG, JPEG veya PNG formatında kolon doku görüntüsü yükle. Model görüntüyü analiz ederek kanser veya normal sınıf olasılıklarını raporlar.",
        "upload_label": "Görüntü yükle",
        "how_title": "Nasıl test edilir?",
        "how_desc": "Kolon histopatoloji örnekleriyle test et. Normal fotoğraflar, röntgenler veya alakasız görseller modelin amaçlanan veri dağılımı dışında kalır.",
        "performance_title": "Model performansı",
        "performance_desc": "EfficientNet-B0 modeli, ResNet18 baseline modeline göre test ayrımında false negative sayısını düşürdüğü için ana model olarak seçildi.",
        "accuracy": "Test doğruluğu",
        "recall": "Kanser recall",
        "fn": "False negative",
        "prediction_result": "Tahmin sonucu",
        "cancer_result": "KANSER / Adenokarsinom benzeri doku",
        "normal_result": "NORMAL / Benign benzeri doku",
        "confidence": "Güven",
        "cancer_prob": "Kanser olasılığı",
        "normal_prob": "Normal olasılığı",
        "curves": "Eğitim grafikleri",
        "curves_desc": "EfficientNet-B0 eğitim sürecinden accuracy, cancer recall ve loss grafikleri.",
        "limitation_title": "Önemli sınırlama",
        "limitation_desc": "LC25000 veri seti artırılmış görüntüler içerir. Bu nedenle test sonuçları gerçek klinik performans olarak yorumlanmamalıdır. Gerçek klinik kullanım için harici doğrulama, uzman değerlendirmesi, regülasyon incelemesi ve farklı cihazlar/boyama protokolleri üzerinde test gerekir.",
        "disclaimer": "Bu uygulama yalnızca araştırma ve eğitim demosudur. Klinik teşhis, tedavi planlama veya tıbbi karar verme amacıyla kullanılmaz. Nihai değerlendirme her zaman uzman patologlar tarafından yapılmalıdır.",
        "analysis_waiting": "Analiz için bir görüntü yükle",
        "analysis_waiting_desc": "Görüntü yüklendiğinde model ön işleme, tensör dönüşümü, sınıf olasılığı ve güven skorunu hesaplar.",
        "uploaded_caption": "Yüklenen histopatoloji görüntüsü",
        "analysis_running": "AI analiz tamamlandı",
        "step_1": "Doku görüntüsünü yükle",
        "step_2": "Ön işleme uygula",
        "step_3": "AI modeli çalıştır",
        "step_4": "Tahmini incele",
        "spinner": "AI patoloji analizi yapılıyor...",
        "website": "Canlı Website",
        "github": "GitHub Repo",
        "footer": "Mehmet Cam tarafından geliştirildi · OncoConnect AI · Araştırma prototipi",
    }
else:
    T = {
        "badge": "OncoConnect AI Research Prototype",
        "title": "Colon Cancer Histopathology AI",
        "subtitle": "An EfficientNet-B0 based research demo designed to distinguish colon adenocarcinoma tissue from benign colon tissue using histopathology images.",
        "upload_title": "Upload a histopathology image",
        "upload_desc": "Upload a JPG, JPEG, or PNG colon tissue image. The model will analyze it and report cancer or normal class probabilities.",
        "upload_label": "Upload image",
        "how_title": "How to test",
        "how_desc": "Use colon histopathology sample images. Regular photos, X-rays, or unrelated images are outside the model's intended distribution.",
        "performance_title": "Model performance",
        "performance_desc": "The EfficientNet-B0 model was selected as the main model because it reduced false negatives compared with the ResNet18 baseline on the test split.",
        "accuracy": "Test accuracy",
        "recall": "Cancer recall",
        "fn": "False negatives",
        "prediction_result": "Prediction result",
        "cancer_result": "CANCER / Adenocarcinoma-like tissue",
        "normal_result": "NORMAL / Benign-like tissue",
        "confidence": "Confidence",
        "cancer_prob": "Cancer probability",
        "normal_prob": "Normal probability",
        "curves": "Training curves",
        "curves_desc": "Accuracy, cancer recall, and loss curves from the EfficientNet-B0 training run.",
        "limitation_title": "Important limitation",
        "limitation_desc": "The LC25000 dataset contains augmented images. Therefore, the reported test results should not be interpreted as real-world clinical performance. Real clinical deployment would require external validation, expert review, regulatory evaluation, and testing across different scanners, staining protocols, and patient populations.",
        "disclaimer": "This application is for research and educational demonstration only. It is not intended for clinical diagnosis, treatment planning, or medical decision-making. Final evaluation must always be performed by qualified pathology specialists.",
        "analysis_waiting": "Upload an image to start analysis",
        "analysis_waiting_desc": "Once an image is uploaded, the model performs preprocessing, tensor conversion, class probability estimation, and confidence scoring.",
        "uploaded_caption": "Uploaded histopathology image",
        "analysis_running": "AI analysis completed",
        "step_1": "Upload tissue image",
        "step_2": "Preprocess input",
        "step_3": "Run AI model",
        "step_4": "Review prediction",
        "spinner": "AI pathology analysis is running...",
        "website": "Live Website",
        "github": "GitHub Repo",
        "footer": "Built by Mehmet Cam · OncoConnect AI · Research prototype",
    }


if is_dark:
    app_bg = """
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.30), transparent 34%),
        radial-gradient(circle at top right, rgba(20, 184, 166, 0.24), transparent 30%),
        linear-gradient(180deg, #020617 0%, #0f172a 48%, #111827 100%);
    """
    text_main = "#f8fafc"
    text_muted = "#cbd5e1"
    card_bg = "rgba(15, 23, 42, 0.84)"
    card_border = "rgba(148, 163, 184, 0.24)"
    soft_card = "rgba(30, 41, 59, 0.78)"
    upload_bg = "#111827"
    disclaimer_text = "#fed7aa"
else:
    app_bg = """
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 45%, #ffffff 100%);
    """
    text_main = "#0f172a"
    text_muted = "#64748b"
    card_bg = "rgba(255, 255, 255, 0.86)"
    card_border = "rgba(148, 163, 184, 0.30)"
    soft_card = "rgba(248, 250, 252, 0.95)"
    upload_bg = "#f8fafc"
    disclaimer_text = "#7c2d12"


html(
    f"""
    <style>
    .stApp {{
        {app_bg}
    }}

    .main .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    .hero-card {{
        padding: 2.35rem;
        border-radius: 30px;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.97) 0%, rgba(30, 58, 138, 0.96) 54%, rgba(15, 118, 110, 0.96) 100%);
        color: white;
        box-shadow: 0 28px 80px rgba(2, 6, 23, 0.38);
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.14);
        overflow: hidden;
        position: relative;
    }}

    .hero-card:before {{
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        border-radius: 999px;
        right: -120px;
        top: -140px;
        background: rgba(45, 212, 191, 0.20);
        filter: blur(4px);
    }}

    .topbar {{
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: center;
        margin-bottom: 1.2rem;
        position: relative;
        z-index: 1;
    }}

    .brand {{
        display: flex;
        gap: 0.75rem;
        align-items: center;
        font-weight: 950;
        letter-spacing: -0.03em;
        font-size: 1.05rem;
    }}

    .logo-mark {{
        width: 44px;
        height: 44px;
        border-radius: 15px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, #60a5fa, #2dd4bf);
        box-shadow: 0 12px 30px rgba(45, 212, 191, 0.28);
        font-size: 1.35rem;
    }}

    .mode-pill {{
        padding: 0.58rem 0.92rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.24);
        color: rgba(255, 255, 255, 0.94);
        font-size: 0.88rem;
        font-weight: 850;
    }}

    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.44rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        font-size: 0.86rem;
        font-weight: 850;
        letter-spacing: 0.02em;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }}

    .hero-title {{
        font-size: 3.05rem;
        line-height: 1.04;
        font-weight: 950;
        margin: 0 0 1rem 0;
        letter-spacing: -0.06em;
        position: relative;
        z-index: 1;
    }}

    .hero-subtitle {{
        font-size: 1.08rem;
        line-height: 1.75;
        color: rgba(255, 255, 255, 0.87);
        max-width: 860px;
        margin-bottom: 1.25rem;
        position: relative;
        z-index: 1;
    }}

    .hero-actions {{
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin: 1.3rem 0 0.6rem 0;
        position: relative;
        z-index: 1;
    }}

    .action-button {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.74rem 1rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.95);
        color: #0f172a !important;
        text-decoration: none !important;
        font-weight: 850;
        border: 1px solid rgba(255, 255, 255, 0.35);
    }}

    .action-button.secondary {{
        background: rgba(255, 255, 255, 0.12);
        color: white !important;
    }}

    .hero-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.25rem;
        position: relative;
        z-index: 1;
    }}

    .meta-pill {{
        padding: 0.62rem 0.9rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.20);
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.92);
        font-weight: 700;
    }}

    .glass-card {{
        padding: 1.35rem;
        border-radius: 24px;
        background: {card_bg};
        border: 1px solid {card_border};
        box-shadow: 0 22px 58px rgba(2, 6, 23, 0.12);
        backdrop-filter: blur(14px);
        margin-bottom: 1rem;
        color: {text_main};
    }}

    .section-title {{
        font-size: 1.28rem;
        font-weight: 900;
        color: {text_main};
        margin-bottom: 0.55rem;
        letter-spacing: -0.035em;
    }}

    .muted {{
        color: {text_muted};
        line-height: 1.7;
        font-size: 0.98rem;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1rem;
    }}

    .metric-card {{
        padding: 1rem;
        border-radius: 18px;
        background: {soft_card};
        border: 1px solid {card_border};
    }}

    .metric-label {{
        color: {text_muted};
        font-size: 0.78rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.065em;
        margin-bottom: 0.35rem;
    }}

    .metric-value {{
        color: {text_main};
        font-size: 1.62rem;
        font-weight: 950;
        letter-spacing: -0.05em;
    }}

    .analysis-steps {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin-top: 1rem;
    }}

    .step {{
        padding: 0.85rem;
        border-radius: 16px;
        background: {soft_card};
        border: 1px solid {card_border};
    }}

    .step-number {{
        color: #38bdf8;
        font-weight: 950;
        margin-bottom: 0.2rem;
    }}

    .step-text {{
        color: {text_muted};
        font-size: 0.84rem;
        font-weight: 700;
    }}

    .prediction-card-cancer {{
        padding: 1.35rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(239, 68, 68, 0.28);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}

    .prediction-card-normal {{
        padding: 1.35rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #047857 0%, #10b981 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(16, 185, 129, 0.28);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}

    .prediction-label {{
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.88;
        font-weight: 900;
    }}

    .prediction-main {{
        font-size: 1.8rem;
        font-weight: 950;
        letter-spacing: -0.045em;
        margin-top: 0.15rem;
    }}

    .prediction-confidence {{
        margin-top: 0.3rem;
        font-size: 1.05rem;
        opacity: 0.93;
        font-weight: 700;
    }}

    .disclaimer {{
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(251, 146, 60, 0.13);
        border: 1px solid rgba(251, 146, 60, 0.45);
        color: {disclaimer_text};
        line-height: 1.65;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        font-weight: 650;
    }}

    .footer-note {{
        text-align: center;
        color: {text_muted};
        font-size: 0.9rem;
        margin-top: 2rem;
    }}

    div[data-testid="stFileUploader"] {{
        padding: 0.75rem;
        border-radius: 20px;
        background: {upload_bg};
        border: 1px dashed rgba(148, 163, 184, 0.65);
    }}

    div[data-testid="stFileUploader"] section {{
        background: transparent;
    }}

    div[data-testid="stMetricValue"] {{
        font-weight: 900;
    }}

    hr {{
        opacity: 0.18;
    }}

    @media (max-width: 768px) {{
        .hero-title {{
            font-size: 2.15rem;
        }}
        .metric-grid,
        .analysis-steps {{
            grid-template-columns: 1fr;
        }}
        .topbar {{
            align-items: flex-start;
            flex-direction: column;
        }}
    }}
    </style>
    """
)


html(
    f"""
    <div class="hero-card">
    <div class="topbar">
    <div class="brand">
    <div class="logo-mark">🧬</div>
    <div>OncoConnect AI</div>
    </div>
    <div class="mode-pill">● {mode}</div>
    </div>
    <div class="hero-badge">✨ {T["badge"]}</div>
    <div class="hero-title">{T["title"]}</div>
    <div class="hero-subtitle">{T["subtitle"]}</div>
    <div class="hero-actions">
    <a class="action-button" href="{ONCOCONNECT_URL}" target="_blank">🌐 {T["website"]}</a>
    <a class="action-button secondary" href="{GITHUB_URL}" target="_blank">💻 {T["github"]}</a>
    </div>
    <div class="hero-meta">
    <div class="meta-pill">Model: EfficientNet-B0</div>
    <div class="meta-pill">Dataset: LC25000 colon subset</div>
    <div class="meta-pill">Task: Binary image classification</div>
    <div class="meta-pill">Framework: PyTorch + Streamlit</div>
    </div>
    </div>
    """
)


html(
    f"""
    <div class="disclaimer">
    <strong>Medical disclaimer:</strong> {T["disclaimer"]}
    </div>
    """
)


left, right = st.columns([1.02, 0.98], gap="large")


with left:
    html(
        f"""
        <div class="glass-card">
        <div class="section-title">{T["upload_title"]}</div>
        <div class="muted">{T["upload_desc"]}</div>
        </div>
        """
    )

    uploaded_file = st.file_uploader(
        T["upload_label"],
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption=T["uploaded_caption"], use_container_width=True)

        with st.spinner(T["spinner"]):
            result = predict(image, model, device)

        html(
            f"""
            <div class="glass-card">
            <div class="section-title">{T["analysis_running"]}</div>
            <div class="analysis-steps">
            <div class="step">
            <div class="step-number">01</div>
            <div class="step-text">Image preprocessing</div>
            </div>
            <div class="step">
            <div class="step-number">02</div>
            <div class="step-text">EfficientNet-B0 inference</div>
            </div>
            <div class="step">
            <div class="step-number">03</div>
            <div class="step-text">Class probability scoring</div>
            </div>
            <div class="step">
            <div class="step-number">04</div>
            <div class="step-text">Risk label output</div>
            </div>
            </div>
            </div>
            """
        )

        if result["label"] == "cancer":
            html(
                f"""
                <div class="prediction-card-cancer">
                <div class="prediction-label">{T["prediction_result"]}</div>
                <div class="prediction-main">{T["cancer_result"]}</div>
                <div class="prediction-confidence">{T["confidence"]}: {result["confidence"] * 100:.2f}%</div>
                </div>
                """
            )
        else:
            html(
                f"""
                <div class="prediction-card-normal">
                <div class="prediction-label">{T["prediction_result"]}</div>
                <div class="prediction-main">{T["normal_result"]}</div>
                <div class="prediction-confidence">{T["confidence"]}: {result["confidence"] * 100:.2f}%</div>
                </div>
                """
            )

        p1, p2 = st.columns(2)

        with p1:
            st.metric(T["cancer_prob"], f"{result['cancer_probability'] * 100:.2f}%")

        with p2:
            st.metric(T["normal_prob"], f"{result['normal_probability'] * 100:.2f}%")

        st.progress(result["confidence"])

    else:
        html(
            f"""
            <div class="glass-card">
            <div class="section-title">{T["analysis_waiting"]}</div>
            <div class="muted">{T["analysis_waiting_desc"]}</div>
            <div class="analysis-steps">
            <div class="step">
            <div class="step-number">01</div>
            <div class="step-text">{T["step_1"]}</div>
            </div>
            <div class="step">
            <div class="step-number">02</div>
            <div class="step-text">{T["step_2"]}</div>
            </div>
            <div class="step">
            <div class="step-number">03</div>
            <div class="step-text">{T["step_3"]}</div>
            </div>
            <div class="step">
            <div class="step-number">04</div>
            <div class="step-text">{T["step_4"]}</div>
            </div>
            </div>
            </div>
            <div class="glass-card">
            <div class="section-title">{T["how_title"]}</div>
            <div class="muted">{T["how_desc"]}</div>
            </div>
            """
        )


with right:
    html(
        f"""
        <div class="glass-card">
        <div class="section-title">{T["performance_title"]}</div>
        <div class="muted">{T["performance_desc"]}</div>
        <div class="metric-grid">
        <div class="metric-card">
        <div class="metric-label">{T["accuracy"]}</div>
        <div class="metric-value">100%</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">{T["recall"]}</div>
        <div class="metric-value">100%</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">{T["fn"]}</div>
        <div class="metric-value">0</div>
        </div>
        </div>
        </div>
        """
    )

    html('<div class="glass-card">')
    st.image(
        "assets/efficientnet_confusion_matrix.png",
        caption="EfficientNet-B0 Confusion Matrix",
        use_container_width=True,
    )
    html("</div>")


st.markdown("---")


html(
    f"""
    <div class="glass-card">
    <div class="section-title">{T["curves"]}</div>
    <div class="muted">{T["curves_desc"]}</div>
    </div>
    """
)


c1, c2, c3 = st.columns(3)


with c1:
    html('<div class="glass-card">')
    st.image("assets/efficientnet_accuracy.png", caption="Accuracy", use_container_width=True)
    html("</div>")


with c2:
    html('<div class="glass-card">')
    st.image("assets/efficientnet_cancer_recall.png", caption="Cancer Recall", use_container_width=True)
    html("</div>")


with c3:
    html('<div class="glass-card">')
    st.image("assets/efficientnet_loss.png", caption="Loss", use_container_width=True)
    html("</div>")


html(
    f"""
    <div class="glass-card">
    <div class="section-title">{T["limitation_title"]}</div>
    <div class="muted">{T["limitation_desc"]}</div>
    </div>
    <div class="footer-note">{T["footer"]}</div>
    """
)