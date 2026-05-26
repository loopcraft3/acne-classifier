import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import streamlit as st
import PIL

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AcneScan · AI Skin Analysis",
    page_icon="🔬",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
}

.stApp {
    background: #0a0a0f;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,255,180,0.1);
    border: 1px solid rgba(99,255,180,0.3);
    color: #63ffb4;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 800;
    line-height: 1.05;
    color: #f0f0f5;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    color: #63ffb4;
}
.hero-sub {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.01em;
    margin-bottom: 0;
}

/* Upload zone */
.upload-zone {
    background: #111118;
    border: 1.5px dashed #2a2a3a;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem 0 1.5rem;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #63ffb4; }

/* Result cards */
.result-card {
    background: #111118;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    border: 1px solid #1e1e2e;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #f0f0f5;
    margin-bottom: 0.25rem;
}
.result-value.normal { color: #63ffb4; }
.result-value.level_0 { color: #fbbf24; }
.result-value.level_1 { color: #f97316; }
.result-value.level_2 { color: #ef4444; }

.confidence-bar-bg {
    background: #1e1e2e;
    border-radius: 100px;
    height: 6px;
    margin-top: 1rem;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #63ffb4, #3be8a0);
    transition: width 0.6s ease;
}

/* Severity grid */
.severity-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-top: 1.5rem;
}
.sev-item {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1rem 0.75rem;
    text-align: center;
}
.sev-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    margin: 0 auto 0.5rem;
}
.sev-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #9ca3af;
}
.sev-prob {
    font-size: 0.85rem;
    font-weight: 500;
    color: #6b7280;
    margin-top: 0.2rem;
}

/* Advice pill */
.advice-pill {
    display: inline-block;
    padding: 0.6rem 1.2rem;
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 1rem;
}

/* Override streamlit button */
.stButton > button {
    background: #63ffb4 !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 0.7rem 2.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    color: #0a0a0f !important;
}

/* Uploaded image */
.stImage img {
    border-radius: 12px;
    border: 1px solid #1e1e2e;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: transparent;
}
[data-testid="stFileUploaderDropzone"] {
    background: #111118 !important;
    border: 1.5px dashed #2a2a3a !important;
    border-radius: 16px !important;
}

/* Divider */
.divider {
    height: 1px;
    background: #1e1e2e;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────────────────────
labels = ['level_0', 'level_1', 'level_2', 'normal']

LABEL_META = {
    'normal':  {'color': '#63ffb4', 'dot': '#63ffb4', 'title': 'Normal Skin',  'advice': '✅ Your skin looks healthy! Maintain your routine.'},
    'level_0': {'color': '#fbbf24', 'dot': '#fbbf24', 'title': 'Mild Acne',    'advice': '💊 Try gentle cleansing and non-comedogenic moisturizer.'},
    'level_1': {'color': '#f97316', 'dot': '#f97316', 'title': 'Moderate Acne','advice': '🧴 Consider OTC benzoyl peroxide or salicylic acid treatments.'},
    'level_2': {'color': '#ef4444', 'dot': '#ef4444', 'title': 'Severe Acne',  'advice': '🏥 We recommend consulting a dermatologist.'},
}

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

def create_fastai_resnet18(num_classes=4):
    backbone = models.resnet18(weights=None)
    body = nn.Sequential(*list(backbone.children())[:-2])
    head = nn.Sequential(
        AdaptiveConcatPool2d(), nn.Flatten(),
        nn.BatchNorm1d(1024), nn.Dropout(p=0.25),
        nn.Linear(1024, 512), nn.ReLU(inplace=True),
        nn.BatchNorm1d(512), nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
    )
    return nn.Sequential(body, head)

@st.cache_resource
def load_model():
    model = create_fastai_resnet18(num_classes=4)
    state = torch.load('./data/models/best_resnet.pth', map_location='cpu', weights_only=False)
    weights = state['model'] if 'model' in state else state
    model.load_state_dict(weights)
    model.eval()
    return model

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🔬 AI-Powered · ResNet-18 · 90% Accuracy</div>
  <div class="hero-title">Acne<span>Scan</span></div>
  <div class="hero-sub">Upload a facial photo and get instant acne severity analysis</div>
</div>
""", unsafe_allow_html=True)

upload_file = st.file_uploader("", label_visibility="collapsed", type=["jpg", "jpeg", "png"])

image = None
if upload_file is not None:
    image_data = upload_file.getvalue()
    image = PIL.Image.open(io.BytesIO(image_data)).convert('RGB')
    st.image(image_data, use_container_width=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
predict_btn = st.button("Analyse Skin →")

if predict_btn:
    if image is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Analysing..."):
            model = load_model()
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0).numpy()
                class_idx = int(np.argmax(probs))

        label = labels[class_idx]
        meta  = LABEL_META[label]
        conf  = float(probs[class_idx]) * 100

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Diagnosis</div>
            <div class="result-value {label}">{meta['title']}</div>
            <div style="color:#6b7280;font-size:0.85rem;margin-top:0.25rem">
                Confidence: <strong style="color:#f0f0f5">{conf:.1f}%</strong>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{conf}%;background:linear-gradient(90deg,{meta['color']},{meta['color']}aa)"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # All class probabilities
        st.markdown("""<div class="result-label" style="margin-top:1.5rem;color:#4b5563;font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;font-weight:600">All Classes</div>""", unsafe_allow_html=True)
        grid_html = '<div class="severity-grid">'
        for i, lbl in enumerate(labels):
            m = LABEL_META[lbl]
            p = float(probs[i]) * 100
            active_border = f"border-color:{m['color']};" if i == class_idx else ""
            grid_html += f"""
            <div class="sev-item" style="{active_border}">
                <div class="sev-dot" style="background:{m['dot']}"></div>
                <div class="sev-name">{lbl.replace('_',' ')}</div>
                <div class="sev-prob">{p:.1f}%</div>
            </div>"""
        grid_html += '</div>'
        st.markdown(grid_html, unsafe_allow_html=True)

        # Advice
        st.markdown(f"""
        <div style="margin-top:1.5rem;background:#111118;border:1px solid #1e1e2e;
                    border-left:3px solid {meta['color']};border-radius:12px;
                    padding:1rem 1.2rem;font-size:0.9rem;color:#9ca3af;">
            {meta['advice']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1rem;font-size:0.75rem;color:#374151;text-align:center">
            ⚠️ This tool is for informational purposes only and does not replace professional medical advice.
        </div>
        """, unsafe_allow_html=True)