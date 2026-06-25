import html as html_escape
import textwrap
import time
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

from model import (
    load_model,
    predict,
    create_gradcam_overlay,
    create_attention_box_overlay,
)

from feedback_store import (
    save_feedback_sample,
    feedback_csv_exists,
    read_feedback_csv_bytes,
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
- Kanser-benzeri doku olasılığı: {result["cancer_probability"] * 100:.2f}%
- Normal/benign-benzeri doku olasılığı: {result["normal_probability"] * 100:.2f}%

Kurallar:
- Klinik teşhis koyma.
- Doktor veya patolog yerine geçme.
- Sonucun sadece araştırma/eğitim demosu olduğunu açıkça belirt.
- Grad-CAM/attention bölgesinin kesin tümör sınırı olmadığını söyle.
- Panik yaratmayan, dikkatli ve anlaşılır bir dil kullan.
- Aşağıdaki 4 başlıkla yaz:
  1. Kısa Özet
  2. Model Çıktısının Yorumu
  3. Güvenlik ve Sınırlamalar
  4. Önerilen Sonraki Adım
- Türkçe yaz.
"""
    else:
        user_prompt = f"""
Create a short, safe, and professional AI interpretation report for a histopathology image classification demo.

Model output:
- Prediction: {result["label"]}
- Confidence: {result["confidence"] * 100:.2f}%
- Cancer-like tissue probability: {result["cancer_probability"] * 100:.2f}%
- Normal/benign-like tissue probability: {result["normal_probability"] * 100:.2f}%

Rules:
- Do not provide a clinical diagnosis.
- Do not replace a doctor or pathology specialist.
- Clearly state that this is a research/educational demo only.
- Explain that Grad-CAM/attention regions are not confirmed tumor boundaries.
- Use calm, careful, and understandable language.
- Write with these 4 headings:
  1. Brief Summary
  2. Interpretation of Model Output
  3. Safety and Limitations
  4. Suggested Next Step
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
        "max_tokens": 900,
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


def make_report_download_text(item, report):
    result = item["result"]
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""OncoConnect AI - Colon Histopathology Research Demo Report
Generated at: {created_at}
File: {item["filename"]}

MODEL OUTPUT
Prediction: {result["label"]}
Confidence: {result["confidence"] * 100:.2f}%
Cancer-like probability: {result["cancer_probability"] * 100:.2f}%
Normal-like probability: {result["normal_probability"] * 100:.2f}%

AI INTERPRETATION REPORT
{report}

IMPORTANT DISCLAIMER
This report is for research and educational demonstration only. It is not intended for clinical diagnosis, treatment planning, or medical decision-making. Final evaluation must always be performed by qualified pathology specialists.
"""


def probability_bar(label, value, color="#38bdf8"):
    pct = max(0, min(100, value * 100))
    html_block(
        f"""
        <div class="prob-wrap">
        <div class="prob-head">
        <span>{label}</span>
        <strong>{pct:.2f}%</strong>
        </div>
        <div class="prob-track">
        <div class="prob-fill" style="width: {pct:.2f}%; background: {color};"></div>
        </div>
        </div>
        """
    )


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

    st.divider()

    if feedback_csv_exists():
        feedback_bytes = read_feedback_csv_bytes()

        if feedback_bytes:
            st.download_button(
                label="⬇️ Download feedback dataset",
                data=feedback_bytes,
                file_name="oncoconnect_feedback_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )


is_tr = lang == "Türkçe"
is_dark = theme == "Dark"


if is_tr:
    T = {
        "badge": "Araştırma prototipi",
        "title": "Kolon Histopatoloji AI Analiz Paneli",
        "subtitle": "EfficientNet-B0, Grad-CAM, OpenRouter o4-mini ve insan denetimli feedback toplama desteğiyle kolon histopatoloji görüntü sınıflandırma demosu.",
        "upload_title": "1. Görüntüleri yükle",
        "upload_desc": "Bir veya birden fazla JPG, JPEG ya da PNG kolon doku görüntüsü yükle. Model yalnızca analiz butonuna bastığında çalışır.",
        "upload_label": "Görüntü yükle",
        "ready": "görüntü yüklendi. Analiz için hazır.",
        "start_analysis": "AI Analizini Başlat",
        "performance_title": "Model kartı",
        "performance_desc": "Bu demo, kolon adenokarsinom-benzeri doku ile benign/normal-benzeri kolon dokusunu ayırt eden bir araştırma prototipidir.",
        "accuracy": "Test doğruluğu",
        "recall": "Kanser recall",
        "fn": "False negative",
        "prediction_result": "Tahmin sonucu",
        "cancer_result": "KANSER-BENZERİ DOKU ÖRÜNTÜSÜ",
        "normal_result": "NORMAL / BENIGN-BENZERİ DOKU ÖRÜNTÜSÜ",
        "confidence": "Güven",
        "cancer_prob": "Kanser-benzeri olasılık",
        "normal_prob": "Normal-benzeri olasılık",
        "curves": "Eğitim grafikleri",
        "curves_desc": "EfficientNet-B0 eğitim sürecinden accuracy, cancer recall ve loss grafikleri.",
        "limitation_title": "Önemli sınırlama",
        "limitation_desc": "LC25000 veri seti artırılmış görüntüler içerir. Bu nedenle test sonuçları gerçek klinik performans olarak yorumlanmamalıdır. Gerçek klinik kullanım için harici doğrulama, uzman değerlendirmesi, regülasyon incelemesi ve farklı cihazlar/boyama protokolleri üzerinde test gerekir.",
        "disclaimer": "Bu uygulama yalnızca araştırma ve eğitim demosudur. Klinik teşhis, tedavi planlama veya tıbbi karar verme amacıyla kullanılmaz. Nihai değerlendirme her zaman uzman patologlar tarafından yapılmalıdır.",
        "analysis_waiting": "Analiz için görüntü bekleniyor",
        "analysis_waiting_desc": "Görüntü yüklendikten sonra AI Analizini Başlat butonuna bas. Model sınıf olasılığı, güven skoru, Grad-CAM açıklama haritası ve yaklaşık dikkat kutusu üretir.",
        "analysis_results": "3. Analiz sonuçları",
        "analysis_results_desc": "Her görüntü için tahmin, olasılıklar, Grad-CAM heatmap ve yaklaşık dikkat kutusu aşağıda gösterilir.",
        "step_1": "Upload",
        "step_2": "Analyze",
        "step_3": "Review",
        "step_1_desc": "Doku görüntüleri yüklenir",
        "step_2_desc": "EfficientNet-B0 ve Grad-CAM çalışır",
        "step_3_desc": "Sonuçlar, rapor ve feedback incelenir",
        "website": "Canlı Website",
        "github": "GitHub Repo",
        "footer": "Mehmet Cam tarafından geliştirildi · OncoConnect AI · Araştırma prototipi",
        "explain_title": "Model attention map",
        "explain_desc": "Bu Grad-CAM görselleştirmesi, modelin tahmine katkı veren bölgelerini gösterir. Bu bir klinik tümör sınırı veya kesin patoloji işaretlemesi değildir.",
        "original": "Orijinal",
        "heatmap": "Grad-CAM",
        "box": "Attention zone",
        "box_note": "Kırmızı kutu, model aktivasyonunun yoğun olduğu yaklaşık alanı gösterir. Klinik sınır/segmentasyon değildir.",
        "ai_report_title": "AI yorum raporu",
        "ai_report_desc": "OpenRouter o4-mini, model çıktısını güvenli ve tıbbi olmayan bir açıklamaya dönüştürür.",
        "ai_report_button": "AI Raporu Oluştur",
        "ai_report_warning": "Bu metin OpenRouter o4-mini ile üretilmiştir. Tıbbi tavsiye veya klinik teşhis olarak değerlendirilmemelidir.",
        "download_report": "Raporu TXT indir",
        "queue": "Analiz kuyruğu",
        "queue_desc": "Yüklenen görüntüler analiz için sıraya alındı.",
        "feedback_title": "Bu araştırma modelini geliştirmeye yardımcı ol",
        "feedback_desc": "Bu feedback, gelecekte insan denetimli inceleme ve kontrollü yeniden eğitim için aday veri olarak kaydedilir. Model bu yüklemeden otomatik öğrenmez.",
        "feedback_question": "Model tahmini faydalı mıydı?",
        "feedback_label": "Opsiyonel etiket / uzman etiketi",
        "feedback_notes": "Opsiyonel not",
        "feedback_button": "Gelecekteki retraining için feedback kaydet",
        "feedback_success": "Feedback kaydedildi. Aday örnek ID:",
    }
else:
    T = {
        "badge": "Research prototype",
        "title": "Colon Histopathology AI Analysis Console",
        "subtitle": "A Streamlit research demo powered by EfficientNet-B0, Grad-CAM explainability, OpenRouter o4-mini reporting, and human-in-the-loop feedback collection.",
        "upload_title": "1. Upload images",
        "upload_desc": "Upload one or more JPG, JPEG, or PNG colon tissue images. The model runs only after you click the analysis button.",
        "upload_label": "Upload image",
        "ready": "image(s) uploaded. Ready for analysis.",
        "start_analysis": "Start AI Analysis",
        "performance_title": "Model card",
        "performance_desc": "This demo classifies colon adenocarcinoma-like tissue patterns versus benign/normal-like colon tissue patterns.",
        "accuracy": "Test accuracy",
        "recall": "Cancer recall",
        "fn": "False negatives",
        "prediction_result": "Prediction result",
        "cancer_result": "CANCER-LIKE TISSUE PATTERN",
        "normal_result": "NORMAL / BENIGN-LIKE TISSUE PATTERN",
        "confidence": "Confidence",
        "cancer_prob": "Cancer-like probability",
        "normal_prob": "Normal-like probability",
        "curves": "Training curves",
        "curves_desc": "Accuracy, cancer recall, and loss curves from the EfficientNet-B0 training run.",
        "limitation_title": "Important limitation",
        "limitation_desc": "The LC25000 dataset contains augmented images. Therefore, the reported test results should not be interpreted as real-world clinical performance. Real clinical deployment would require external validation, expert review, regulatory evaluation, and testing across different scanners, staining protocols, and patient populations.",
        "disclaimer": "This application is for research and educational demonstration only. It is not intended for clinical diagnosis, treatment planning, or medical decision-making. Final evaluation must always be performed by qualified pathology specialists.",
        "analysis_waiting": "Waiting for images",
        "analysis_waiting_desc": "After uploading images, click Start AI Analysis. The model will produce class probabilities, confidence score, Grad-CAM explanation, and approximate attention box.",
        "analysis_results": "3. Analysis results",
        "analysis_results_desc": "Review prediction, probabilities, Grad-CAM heatmap, and approximate attention box for each analyzed image.",
        "step_1": "Upload",
        "step_2": "Analyze",
        "step_3": "Review",
        "step_1_desc": "Upload tissue images",
        "step_2_desc": "Run EfficientNet-B0 + Grad-CAM",
        "step_3_desc": "Review results, report, and feedback",
        "website": "Live Website",
        "github": "GitHub Repo",
        "footer": "Built by Mehmet Cam · OncoConnect AI · Research prototype",
        "explain_title": "Model attention map",
        "explain_desc": "This Grad-CAM visualization shows the regions that contributed most to the model prediction. It is not a clinical tumor boundary or confirmed pathology annotation.",
        "original": "Original",
        "heatmap": "Grad-CAM",
        "box": "Attention zone",
        "box_note": "The red box approximates the most activated model-attention region. It is not a clinical segmentation boundary.",
        "ai_report_title": "AI interpretation report",
        "ai_report_desc": "OpenRouter o4-mini turns the model output into a safe, non-diagnostic explanation.",
        "ai_report_button": "Generate AI Report",
        "ai_report_warning": "This text is generated by OpenRouter o4-mini. It should not be treated as medical advice or a clinical diagnosis.",
        "download_report": "Download TXT report",
        "queue": "Analysis queue",
        "queue_desc": "Uploaded images are queued for manual analysis.",
        "feedback_title": "Help improve this research model",
        "feedback_desc": "Your feedback will be saved as candidate data for future human review and controlled retraining. The model does not learn automatically from this upload.",
        "feedback_question": "Was the model prediction useful?",
        "feedback_label": "Optional label / expert label",
        "feedback_notes": "Optional notes",
        "feedback_button": "Submit feedback for future retraining",
        "feedback_success": "Feedback saved. Candidate sample ID:",
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
    report_bg = "rgba(255, 255, 255, 0.94)"
    report_text = "#111827"


html_block(
    f"""
    <style>
    .stApp {{
        {app_bg}
    }}

    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    .hero-card {{
        padding: 1.55rem;
        border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.97) 0%, rgba(30, 58, 138, 0.96) 56%, rgba(15, 118, 110, 0.96) 100%);
        color: white;
        box-shadow: 0 24px 70px rgba(2, 6, 23, 0.34);
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.14);
        overflow: hidden;
        position: relative;
    }}

    .topbar {{
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: center;
        margin-bottom: 0.8rem;
        position: relative;
        z-index: 1;
    }}

    .brand {{
        display: flex;
        gap: 0.7rem;
        align-items: center;
        font-weight: 950;
        letter-spacing: -0.03em;
        font-size: 1rem;
    }}

    .logo-mark {{
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, #60a5fa, #2dd4bf);
        box-shadow: 0 12px 30px rgba(45, 212, 191, 0.28);
        font-size: 1.25rem;
    }}

    .mode-pill {{
        padding: 0.52rem 0.86rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.24);
        color: rgba(255, 255, 255, 0.94);
        font-size: 0.84rem;
        font-weight: 850;
    }}

    .hero-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.34rem 0.68rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        font-size: 0.78rem;
        font-weight: 850;
        margin-bottom: 0.6rem;
    }}

    .hero-title {{
        font-size: 2.15rem;
        line-height: 1.08;
        font-weight: 950;
        margin: 0 0 0.55rem 0;
        letter-spacing: -0.055em;
    }}

    .hero-subtitle {{
        font-size: 0.98rem;
        line-height: 1.65;
        color: rgba(255, 255, 255, 0.87);
        max-width: 900px;
        margin-bottom: 0.8rem;
    }}

    .hero-actions {{
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin: 0.9rem 0 0.45rem 0;
    }}

    .action-button {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.62rem 0.88rem;
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
        gap: 0.55rem;
        margin-top: 0.85rem;
    }}

    .meta-pill {{
        padding: 0.48rem 0.72rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.20);
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.92);
        font-weight: 700;
    }}

    .stepper {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-bottom: 1rem;
    }}

    .stepper-card {{
        padding: 1rem;
        border-radius: 22px;
        background: {card_bg};
        border: 1px solid {card_border};
        box-shadow: 0 18px 48px rgba(2, 6, 23, 0.10);
    }}

    .stepper-num {{
        color: #38bdf8;
        font-weight: 950;
        font-size: 0.88rem;
        margin-bottom: 0.2rem;
    }}

    .stepper-title {{
        color: {text_main};
        font-weight: 950;
        font-size: 1.06rem;
        margin-bottom: 0.25rem;
    }}

    .stepper-desc {{
        color: {text_muted};
        font-size: 0.88rem;
        line-height: 1.55;
    }}

    .glass-card {{
        padding: 1.25rem;
        border-radius: 24px;
        background: {card_bg};
        border: 1px solid {card_border};
        box-shadow: 0 22px 58px rgba(2, 6, 23, 0.12);
        backdrop-filter: blur(14px);
        margin-bottom: 1rem;
        color: {text_main};
    }}

    .section-title {{
        font-size: 1.22rem;
        font-weight: 950;
        color: {text_main};
        margin-bottom: 0.55rem;
        letter-spacing: -0.035em;
    }}

    .muted {{
        color: {text_muted};
        line-height: 1.7;
        font-size: 0.96rem;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }}

    .metric-card {{
        padding: 0.95rem;
        border-radius: 18px;
        background: {soft_card};
        border: 1px solid {card_border};
    }}

    .metric-label {{
        color: {text_muted};
        font-size: 0.76rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.065em;
        margin-bottom: 0.35rem;
    }}

    .metric-value {{
        color: {text_main};
        font-size: 1.45rem;
        font-weight: 950;
        letter-spacing: -0.05em;
    }}

    .queue-item {{
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        background: {soft_card};
        border: 1px solid {card_border};
        margin-bottom: 0.55rem;
        color: {text_main};
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: center;
    }}

    .queue-badge {{
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.16);
        color: #38bdf8;
        font-weight: 850;
        font-size: 0.78rem;
        white-space: nowrap;
    }}

    .prediction-card-cancer {{
        padding: 1.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(239, 68, 68, 0.28);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}

    .prediction-card-normal {{
        padding: 1.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #047857 0%, #10b981 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(16, 185, 129, 0.28);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}

    .prediction-label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.88;
        font-weight: 900;
    }}

    .prediction-main {{
        font-size: 1.45rem;
        font-weight: 950;
        letter-spacing: -0.045em;
        margin-top: 0.15rem;
    }}

    .prediction-confidence {{
        margin-top: 0.3rem;
        font-size: 1rem;
        opacity: 0.93;
        font-weight: 700;
    }}

    .prob-wrap {{
        margin: 0.85rem 0;
    }}

    .prob-head {{
        display: flex;
        justify-content: space-between;
        color: {text_main};
        font-size: 0.92rem;
        margin-bottom: 0.38rem;
    }}

    .prob-track {{
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.22);
        overflow: hidden;
    }}

    .prob-fill {{
        height: 12px;
        border-radius: 999px;
    }}

    .disclaimer {{
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: rgba(251, 146, 60, 0.13);
        border: 1px solid rgba(251, 146, 60, 0.45);
        color: {disclaimer_text};
        line-height: 1.65;
        margin-bottom: 1rem;
        font-weight: 650;
    }}

    .report-card {{
        padding: 1.25rem;
        border-radius: 22px;
        background: {report_bg};
        border: 1px solid {card_border};
        color: {report_text} !important;
        line-height: 1.85;
        font-size: 0.98rem;
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
            font-size: 1.85rem;
        }}

        .metric-grid,
        .stepper {{
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
    <div class="meta-pill">EfficientNet-B0</div>
    <div class="meta-pill">Grad-CAM</div>
    <div class="meta-pill">OpenRouter o4-mini</div>
    <div class="meta-pill">Human-in-the-loop feedback</div>
    <div class="meta-pill">PyTorch + Streamlit</div>
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


html_block(
    f"""
    <div class="stepper">
    <div class="stepper-card">
    <div class="stepper-num">01</div>
    <div class="stepper-title">{T["step_1"]}</div>
    <div class="stepper-desc">{T["step_1_desc"]}</div>
    </div>
    <div class="stepper-card">
    <div class="stepper-num">02</div>
    <div class="stepper-title">{T["step_2"]}</div>
    <div class="stepper-desc">{T["step_2_desc"]}</div>
    </div>
    <div class="stepper-card">
    <div class="stepper-num">03</div>
    <div class="stepper-title">{T["step_3"]}</div>
    <div class="stepper-desc">{T["step_3_desc"]}</div>
    </div>
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

        html_block(
            f"""
            <div class="glass-card">
            <div class="section-title">{T["queue"]}</div>
            <div class="muted">{T["queue_desc"]}</div>
            </div>
            """
        )

        for idx, uploaded_file in enumerate(uploaded_files):
            html_block(
                f"""
                <div class="queue-item">
                <span>{idx + 1}. {uploaded_file.name}</span>
                <span class="queue-badge">Ready</span>
                </div>
                """
            )

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

        run_analysis = st.button(
            T["start_analysis"],
            type="primary",
            use_container_width=True,
        )

        if run_analysis:
            all_results = []

            progress = st.progress(0)
            status_box = st.empty()

            status_box.markdown(
                """
                <div class="analysis-status">
                Step 1/4 · Preparing uploaded images...
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
                    Step 2/4 · Running EfficientNet-B0 · Image {file_index + 1}/{len(uploaded_files)}
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
                    Step 3/4 · Generating Grad-CAM · Image {file_index + 1}/{len(uploaded_files)}
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
                """
                <div class="analysis-status">
                Step 4/4 · Analysis completed. Results are ready.
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

                metric_cols = st.columns(4)

                with metric_cols[0]:
                    st.metric("Prediction", result["label"])

                with metric_cols[1]:
                    st.metric(T["confidence"], f"{result['confidence'] * 100:.2f}%")

                with metric_cols[2]:
                    st.metric(T["cancer_prob"], f"{result['cancer_probability'] * 100:.2f}%")

                with metric_cols[3]:
                    st.metric(T["normal_prob"], f"{result['normal_probability'] * 100:.2f}%")

                probability_bar(T["cancer_prob"], result["cancer_probability"], "#ef4444")
                probability_bar(T["normal_prob"], result["normal_probability"], "#10b981")

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
                    st.image(image, caption=T["original"], use_container_width=True)

                with g2:
                    st.image(overlay_image, caption=T["heatmap"], use_container_width=True)

                with g3:
                    st.image(boxed_image, caption=T["box"], use_container_width=True)

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
                    <div class="section-title">{T["feedback_title"]}</div>
                    <div class="muted">{T["feedback_desc"]}</div>
                    </div>
                    """
                )

                feedback_key = f"user_feedback_{idx}"
                label_key = f"user_label_{idx}"
                notes_key = f"user_notes_{idx}"
                submit_key = f"submit_feedback_{idx}"

                user_feedback = st.radio(
                    T["feedback_question"],
                    ["Correct", "Incorrect", "Unsure"],
                    horizontal=True,
                    key=feedback_key,
                )

                user_label = st.selectbox(
                    T["feedback_label"],
                    [
                        "not_sure",
                        "cancer_like",
                        "normal_like",
                        "not_histopathology",
                    ],
                    key=label_key,
                )

                notes = st.text_area(
                    T["feedback_notes"],
                    placeholder="Example: image quality is low, unsure about label, needs expert review...",
                    key=notes_key,
                )

                if st.button(
                    T["feedback_button"],
                    type="secondary",
                    use_container_width=True,
                    key=submit_key,
                ):
                    saved_row = save_feedback_sample(
                        image=image,
                        original_filename=item["filename"],
                        result=result,
                        user_feedback=user_feedback,
                        user_label=user_label,
                        notes=notes,
                    )

                    st.success(f"{T['feedback_success']} {saved_row['image_id']}")

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

                    report_text = make_report_download_text(
                        item=item,
                        report=st.session_state[report_key],
                    )

                    st.download_button(
                        label=T["download_report"],
                        data=report_text,
                        file_name=f"oncoconnect_ai_report_{idx + 1}.txt",
                        mime="text/plain",
                        use_container_width=True,
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

    html_block(
        """
        <div class="glass-card">
        <div class="section-title">Technical model card</div>
        <div class="muted">
        <strong>Architecture:</strong> EfficientNet-B0<br>
        <strong>Dataset:</strong> LC25000 colon subset<br>
        <strong>Task:</strong> Binary classification<br>
        <strong>Input:</strong> Histopathology image<br>
        <strong>Output:</strong> cancer-like / normal-like<br>
        <strong>Explainability:</strong> Grad-CAM<br>
        <strong>AI report:</strong> OpenRouter o4-mini<br>
        <strong>Feedback:</strong> Human-in-the-loop candidate data<br>
        <strong>Clinical use:</strong> Not approved
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