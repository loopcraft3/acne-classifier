# 🔬 AcneScan — AI Skin Analysis

> Deep learning web app that classifies acne severity from facial images in seconds.

🌐 **Live App**: [acne-classifier.streamlit.app](https://acne-classifier.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?style=flat-square) ![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange?style=flat-square) ![Accuracy](https://img.shields.io/badge/Accuracy-90%25-brightgreen?style=flat-square)

---

## 📌 About

Acne affects the majority of teenagers and young adults, yet most people lack guidance on how severe their condition is or what steps to take. AcneScan uses a ResNet-18 deep learning model to classify acne severity from a facial image into four categories — giving users instant, actionable feedback.

---

## 🧠 Model & Dataset

| Detail | Info |
|--------|------|
| Architecture | ResNet-18 (transfer learning) |
| Dataset | 250 HD images × 4 classes = 1,000 total |
| Data source | Hand-picked from internet sources |
| Accuracy | ~90% |
| Input size | 224 × 224 px |

### Classes

| Class | Description |
|-------|-------------|
| 🟢 Normal | No significant acne |
| 🟡 Level 0 | Mild acne |
| 🟠 Level 1 | Moderate acne |
| 🔴 Level 2 | Severe acne |

### Data Augmentation
Horizontal & vertical flips · 90° rotations · Random crop (0–50% zoom) · Rotation ±15° · Blur up to 10px

---

## 🖥️ App Features

- Upload any facial photo (JPG/PNG)
- Instant acne severity classification
- Confidence score with visual progress bar
- All 4 class probabilities shown in a grid
- Tailored skincare advice per severity level
- Clean dark-mode UI

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/loopcraft3/acne-classifier.git
cd acne-classifier
```

**2. Create environment (Anaconda Prompt)**
```bash
conda create -n acne-app python=3.9 -y
conda activate acne-app
```

**3. Install dependencies**
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.9.0+cpu torchvision==0.24.0+cpu streamlit altair==4.2.0 Pillow numpy
```

**4. Run**
```bash
streamlit run main.py
```

Open `http://localhost:8501`

---

## 🗂️ Project Structure

```
acne-classifier/
├── main.py                  # Streamlit app (UI + inference)
├── requirements.txt         # Deployment dependencies
├── runtime.txt              # Python version for Streamlit Cloud
├── README.md
└── data/
    ├── models/
    │   └── best_resnet.pth  # Trained model weights (~49MB)
    ├── level_0/             # Training images
    ├── level_1/
    ├── level_2/
    └── normal/
```

---

## 🔮 Future Improvements

- [ ] Larger and more diverse dataset (cross-geographical skin samples)
- [ ] Object detection to localise acne regions on the face
- [ ] More granular severity levels
- [ ] Treatment and product recommendations
- [ ] Virtual dermatologist consultation feature
- [ ] Native mobile app (iOS & Android)
- [ ] Community platform for skincare discussions

---

## ⚠️ Disclaimer

This tool is for informational purposes only and does not replace professional medical advice. Please consult a dermatologist for diagnosis and treatment.

---

## 📄 License

MIT License — free to use and modify.