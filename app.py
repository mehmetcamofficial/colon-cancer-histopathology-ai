import html as html_escape
import textwrap
import time

import requests
import streamlit as st
from PIL import Image

from model import (
    load_model,
    predict,
    create_gradcam_overlay,
    create_attention_box_overlay,
)


GITHUB_URL = "https://github.com/mehmetcamofficial/colon-cancer-histopathology-ai"
ONCOCONNECT_URL = "https://oncoconnectai.com.tr/"


st.set_page_config(
    page_title="OncoConnect AI | Colon Pathology Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def html_block(content: str):
    cleaned = "\n".join(
        line.strip()
        for line in textwrap.dedent(content).strip().splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


@st.cache_resource
def cached_model():
    return load_model()


def generate_ai_report(result, language="English"):
    api_key = st.secrets.get("OPENROUTER_API_KEY", None)

    if not api_key:
        return None, (
            "OPENROUTER_API_KEY secret is missing. "
            "Add it in Streamlit Cloud > App > Settings > Secrets."
        )

    if language == "Türkçe":
        user_prompt = f"""
Bir histopatoloji görüntü sınıflandırma demosu için kısa, güvenli ve profesyonel bir AI yorum raporu hazırla.

Model çıktısı:
- Tahmin: {result["label"]}
- Güven: {result["confidence"] * 100:.2f}%
- Kanser olasılığı: {result["cancer_probability"] * 100:.2f}%
- Normal olasılığı: {result["normal_probability"] * 100:.2f}%

Kurallar:
- Klinik teşhis koyma.
- Doktor veya patolog yerine geçme.
- Sonucun sadece araştırma/eğitim demosu olduğunu açıkça belirt.
- Grad-CAM/attention bölgesinin kesin tümör sınırı olmadığını söyle.
- Panik yaratmayan, dikkatli ve anlaşılır bir dil kullan.
- 4 bölüm yaz:
  1. Kısa özet
  2. Model çıktısının yorumu
  3. Güvenlik ve sınırlamalar
  4. Önerilen sonraki adım
- Türkçe yaz.
"""
    else:
        user_prompt = f"""
Create a short, safe, and professional AI interpretation report for a histopathology image classification demo.

Model output:
- Prediction: {result["label"]}
- Confidence: {result["confidence"] * 100:.2f}%
- Cancer probability: {result["cancer_probability"] * 100:.2f}%
- Normal probability: {result["normal_probability"] * 100:.2f}%

Rules:
- Do not provide a clinical diagnosis.
- Do not replace a doctor or pathology specialist.
- Clearly state that this is a research/educational demo only.
- Explain that Grad-CAM/attention regions are not confirmed tumor boundaries.
- Use calm, careful, and understandable language.
- Write 4 sections:
  1. Brief summary
  2. Interpretation of model output
  3. Safety and limitations
  4. Suggested next step
- Write in English.
"""

    payload = {
        "model": "openai/o4-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cautious medical AI assistant for a research demo. "
                    "You must not diagnose. You explain model outputs safely, clearly, "
                    "and with strong medical limitations."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": ONCOCONNECT_URL,
        "X-Title": "OncoConnect AI Colon Pathology Demo",
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            return None, f"OpenRouter error {response.status_code}: {response.text}"

        data = response.json()
        report = data["choices"][0]["message"]["content"]
        return report, None

    except Exception as e:
        return None, str(e)


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
        "upload_desc": "Bir veya birden fazla JPG, JPEG ya da PNG formatında kolon doku görüntüsü yükle. Model, sen analiz butonuna bastıktan sonra çalışır.",
        "upload_label": "Görüntü yükle",
        "ready": "görüntü yüklendi. AI analizi için hazır.",
        "start_analysis": "AI Analizi Başlat",
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
        "analysis_waiting": "Analiz için görüntü yükle",
        "analysis_waiting_desc": "Görüntü yüklendikten sonra AI Analizi Başlat butonuna bas. Model sınıf olasılığı, güven skoru, Grad-CAM açıklama haritası ve yaklaşık dikkat kutusu üretir.",
        "uploaded_caption": "Yüklenen histopatoloji görüntüsü",
        "analysis_results": "Analiz sonuçları",
        "analysis_results_desc": "Her görüntü için tahmin, olasılıklar, Grad-CAM heatmap ve yaklaşık dikkat kutusu aşağıda gösterilir.",
        "step_1": "Görüntüler hazırlanıyor",
        "step_2": "EfficientNet-B0 çalışıyor",
        "step_3": "Grad-CAM üretiliyor",
        "step_4": "Sonuçlar hazırlanıyor",
        "website": "Canlı Website",
        "github": "GitHub Repo",
        "footer": "Mehmet Cam tarafından geliştirildi · OncoConnect AI · Araştırma prototipi",
        "explain_title": "AI tarafından öne çıkarılan bölge",
        "explain_desc": "Bu Grad-CAM görselleştirmesi, modelin tahmine katkı veren bölgelerini gösterir. Bu bir klinik tümör sınırı veya kesin patoloji işaretlemesi değildir.",
        "original": "Orijinal görüntü",
        "heatmap": "Grad-CAM heatmap",
        "box": "Yaklaşık dikkat kutusu",
        "box_note": "Kırmızı kutu, model aktivasyonunun yoğun olduğu yaklaşık alanı gösterir. Klinik sınır/segmentasyon değildir.",
        "ai_report_title": "AI yorum raporu",
        "ai_report_desc": "OpenRouter o4-mini, model çıktısını güvenli ve tıbbi olmayan bir açıklamaya dönüştürür.",
        "ai_report_button": "AI Raporu Oluştur",
        "ai_report_warning": "Bu metin OpenRouter o4-mini ile üretilmiştir. Tıbbi tavsiye veya klinik teşhis olarak değerlendirilmemelidir.",
        "no_image": "Henüz görüntü yüklenmedi.",
    }
else:
    T = {
        "badge": "OncoConnect AI Research Prototype",
        "title": "Colon Cancer Histopathology AI",
        "subtitle": "An EfficientNet-B0 based research demo designed to distinguish colon adenocarcinoma tissue from benign colon tissue using histopathology images.",
        "upload_title": "Upload histopathology images",
        "upload_desc": "Upload one or more JPG, JPEG, or PNG colon tissue images. The model will run only after you click the analysis button.",
        "upload_label": "Upload image",
        "ready": "image(s) uploaded. Ready for AI analysis.",
        "start_analysis": "Start AI Analysis",
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
        "analysis_waiting": "Upload images to start analysis",
        "analysis_waiting_desc": "After uploading images, click Start AI Analysis. The model will produce class probabilities, confidence score, Grad-CAM explanation, and approximate attention box.",
        "uploaded_caption": "Uploaded histopathology image",
        "analysis_results": "Analysis results",
        "analysis_results_desc": "Review prediction, probabilities, Grad-CAM heatmap, and approximate attention box for each analyzed image.",
        "step_1": "Preparing images",
        "step_2": "Running EfficientNet-B0",
        "step_3": "Generating Grad-CAM",
        "step_4": "Finalizing results",
        "website": "Live Website",
        "github": "GitHub Repo",
        "footer": "Built by Mehmet Cam · OncoConnect AI · Research prototype",
        "explain_title": "AI-highlighted region",
        "explain_desc": "This Grad-CAM visualization shows the regions that contributed most to the model prediction. It is not a clinical tumor boundary or confirmed pathology annotation.",
        "original": "Original image",
        "heatmap": "Grad-CAM heatmap",
        "box": "Approximate attention box",
        "box_note": "The red box approximates the most activated model-attention region. It is not a clinical segmentation boundary.",
        "ai_report_title": "AI interpretation report",
        "ai_report_desc": "OpenRouter o4-mini turns the model output into a safe, non-diagnostic explanation.",
        "ai_report_button": "Generate AI Report",
        "ai_report_warning": "This text is generated by OpenRouter o4-mini. It should not be treated as medical advice or a clinical diagnosis.",
        "no_image": "No image uploaded yet.",
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
    report_bg = "rgba(255, 255, 255, 0.08)"
    report_text = "#e5e7eb"
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
    report_bg = "rgba(255, 255, 255, 0.92)"
    report_text = "#111827"


html_block(
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

    .report-card {{
        padding: 1.35rem;
        border-radius: 24px;
        background: {report_bg};
        border: 1px solid {card_border};
        color: {report_text} !important;
        line-height: 1.85;
        font-size: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        white-space: pre-wrap;
    }}

    .analysis-status {{
        padding: 1rem;
        border-radius: 18px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: {"#dbeafe" if is_dark else "#0f172a"};
        margin-bottom: 1rem;
        font-weight: 750;
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

    div.stButton > button {{
        border-radius: 999px;
        padding: 0.7rem 1.2rem;
        font-weight: 850;
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


html_block(
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
    <div class="meta-pill">AI Report: OpenRouter o4-mini</div>
    <div class="meta-pill">Explainability: Grad-CAM</div>
    <div class="meta-pill">Framework: PyTorch + Streamlit</div>
    </div>
    </div>
    """
)


html_block(
    f"""
    <div class="disclaimer">
    <strong>Medical disclaimer:</strong> {T["disclaimer"]}
    </div>
    """
)


left, right = st.columns([1.02, 0.98], gap="large")


with left:
    html_block(
        f"""
        <div class="glass-card">
        <div class="section-title">{T["upload_title"]}</div>
        <div class="muted">{T["upload_desc"]}</div>
        </div>
        """
    )

    uploaded_files = st.file_uploader(
        T["upload_label"],
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} {T['ready']}")

        preview_count = min(len(uploaded_files), 3)
        preview_cols = st.columns(preview_count)

        for idx, uploaded_file in enumerate(uploaded_files[:preview_count]):
            uploaded_file.seek(0)
            preview_image = Image.open(uploaded_file).convert("RGB")
            with preview_cols[idx]:
                st.image(
                    preview_image,
                    caption=f"Preview {idx + 1}",
                    use_container_width=True,
                )

        html_block(
            f"""
            <div class="glass-card">
            <div class="section-title">AI analysis control</div>
            <div class="muted">
            Images are uploaded but the model has not run yet. Click the button below to start classification,
            Grad-CAM explanation, attention box generation, and optional OpenRouter AI reporting.
            </div>
            </div>
            """
        )

        run_analysis = st.button(
            T["start_analysis"],
            type="primary",
            use_container_width=True,
        )

        if run_analysis:
            all_results = []

            progress = st.progress(0)
            status_box = st.empty()

            steps = [
                T["step_1"],
                T["step_2"],
                T["step_3"],
                T["step_4"],
            ]

            status_box.markdown(
                f"""
                <div class="analysis-status">
                Step 1/4 · {steps[0]}
                </div>
                """,
                unsafe_allow_html=True,
            )
            progress.progress(10)
            time.sleep(0.25)

            for file_index, uploaded_file in enumerate(uploaded_files):
                uploaded_file.seek(0)
                image = Image.open(uploaded_file).convert("RGB")

                status_box.markdown(
                    f"""
                    <div class="analysis-status">
                    Step 2/4 · {steps[1]} · Image {file_index + 1}/{len(uploaded_files)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                progress.progress(35)
                time.sleep(0.25)

                result = predict(image, model, device)

                status_box.markdown(
                    f"""
                    <div class="analysis-status">
                    Step 3/4 · {steps[2]} · Image {file_index + 1}/{len(uploaded_files)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                progress.progress(65)
                time.sleep(0.25)

                overlay_image, heatmap_image, cam_map = create_gradcam_overlay(
                    image,
                    model,
                    device,
                    class_idx=result["class_index"],
                )

                boxed_image = create_attention_box_overlay(
                    image,
                    cam_map,
                    threshold=0.55,
                )

                all_results.append(
                    {
                        "filename": uploaded_file.name,
                        "image": image,
                        "result": result,
                        "overlay_image": overlay_image,
                        "heatmap_image": heatmap_image,
                        "boxed_image": boxed_image,
                    }
                )

            status_box.markdown(
                f"""
                <div class="analysis-status">
                Step 4/4 · {steps[3]} · Analysis completed.
                </div>
                """,
                unsafe_allow_html=True,
            )
            progress.progress(100)
            time.sleep(0.25)

            st.session_state["analysis_results"] = all_results

    else:
        html_block(
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
            """
        )

    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]

        html_block(
            f"""
            <div class="glass-card">
            <div class="section-title">{T["analysis_results"]}</div>
            <div class="muted">{T["analysis_results_desc"]}</div>
            </div>
            """
        )

        for idx, item in enumerate(results):
            result = item["result"]
            image = item["image"]
            overlay_image = item["overlay_image"]
            boxed_image = item["boxed_image"]

            with st.expander(f"Image {idx + 1}: {item['filename']}", expanded=True):
                if result["label"] == "cancer":
                    html_block(
                        f"""
                        <div class="prediction-card-cancer">
                        <div class="prediction-label">{T["prediction_result"]}</div>
                        <div class="prediction-main">{T["cancer_result"]}</div>
                        <div class="prediction-confidence">{T["confidence"]}: {result["confidence"] * 100:.2f}%</div>
                        </div>
                        """
                    )
                else:
                    html_block(
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

                html_block(
                    f"""
                    <div class="glass-card">
                    <div class="section-title">{T["explain_title"]}</div>
                    <div class="muted">{T["explain_desc"]}</div>
                    </div>
                    """
                )

                g1, g2, g3 = st.columns(3)

                with g1:
                    st.image(
                        image,
                        caption=T["original"],
                        use_container_width=True,
                    )

                with g2:
                    st.image(
                        overlay_image,
                        caption=T["heatmap"],
                        use_container_width=True,
                    )

                with g3:
                    st.image(
                        boxed_image,
                        caption=T["box"],
                        use_container_width=True,
                    )

                html_block(
                    f"""
                    <div class="glass-card">
                    <div class="muted">{T["box_note"]}</div>
                    </div>
                    """
                )

                html_block(
                    f"""
                    <div class="glass-card">
                    <div class="section-title">{T["ai_report_title"]}</div>
                    <div class="muted">{T["ai_report_desc"]}</div>
                    </div>
                    """
                )

                report_key = f"ai_report_{idx}"

                if st.button(
                    T["ai_report_button"],
                    type="primary",
                    use_container_width=True,
                    key=f"report_button_{idx}",
                ):
                    report_progress = st.progress(0)
                    report_status = st.empty()

                    report_status.markdown(
                        """
                        <div class="analysis-status">
                        AI report generation · Preparing model output...
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    report_progress.progress(25)
                    time.sleep(0.25)

                    report_status.markdown(
                        """
                        <div class="analysis-status">
                        AI report generation · Sending safe prompt to OpenRouter o4-mini...
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    report_progress.progress(60)

                    report, error = generate_ai_report(result, language=lang)

                    report_status.markdown(
                        """
                        <div class="analysis-status">
                        AI report generation · Finalizing explanation...
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    report_progress.progress(90)
                    time.sleep(0.25)

                    if error:
                        st.error(error)
                    else:
                        st.session_state[report_key] = report

                    report_progress.progress(100)
                    report_status.empty()

                if report_key in st.session_state:
                    html_block(
                        f"""
                        <div class="glass-card">
                        <div class="section-title">{T["ai_report_title"]}</div>
                        <div class="muted">{T["ai_report_warning"]}</div>
                        </div>
                        """
                    )

                    safe_report = html_escape.escape(st.session_state[report_key])
                    html_block(
                        f"""
                        <div class="report-card">
                        {safe_report}
                        </div>
                        """
                    )


with right:
    html_block(
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

    html_block('<div class="glass-card">')
    st.image(
        "assets/efficientnet_confusion_matrix.png",
        caption="EfficientNet-B0 Confusion Matrix",
        use_container_width=True,
    )
    html_block("</div>")


st.markdown("---")


html_block(
    f"""
    <div class="glass-card">
    <div class="section-title">{T["curves"]}</div>
    <div class="muted">{T["curves_desc"]}</div>
    </div>
    """
)


c1, c2, c3 = st.columns(3)


with c1:
    html_block('<div class="glass-card">')
    st.image("assets/efficientnet_accuracy.png", caption="Accuracy", use_container_width=True)
    html_block("</div>")


with c2:
    html_block('<div class="glass-card">')
    st.image("assets/efficientnet_cancer_recall.png", caption="Cancer Recall", use_container_width=True)
    html_block("</div>")


with c3:
    html_block('<div class="glass-card">')
    st.image("assets/efficientnet_loss.png", caption="Loss", use_container_width=True)
    html_block("</div>")


html_block(
    f"""
    <div class="glass-card">
    <div class="section-title">{T["limitation_title"]}</div>
    <div class="muted">{T["limitation_desc"]}</div>
    </div>
    <div class="footer-note">{T["footer"]}</div>
    """
)