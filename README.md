# 🔍 XPURL — Explainable Phishing URL Detection System

An intelligent and explainable phishing detection system that combines **deep learning**, **statistical feature engineering**, and **model interpretability (SHAP)** to accurately classify URLs as **PHISHING** or **LEGITIMATE**.

---

## 🚀 Overview

XPURL is designed to detect modern phishing attacks using a hybrid approach:

* 🔡 **Character-level semantic encoding** (DistilBERT)
* 📊 **Statistical feature extraction**
* 🔍 **Explainability using SHAP**
* 🔁 **Continual learning–ready architecture (EWC support)**

It is deployed as a **Flask API with a frontend UI** for easy interaction.

---

## ✨ Key Features

* ✅ Hybrid deep learning + handcrafted features
* ✅ Real-time phishing detection
* ✅ SHAP-based explanation of predictions
* ✅ Batch URL prediction support
* ✅ Demo mode (works without model)
* ✅ REST API for integration

---

## 🧠 Model Architecture

* **DistilBERT** → learns URL patterns
* **Statistical Feature Network** → detects structural anomalies
* **Fusion Layer** → combines both
* **Classifier** → predicts phishing probability

---

## 📂 Project Structure

```
xpurl/
│
├── app.py
├── index.html
├── xpurl_saved/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚠️ Model Files (Download Required)

Large model files are not included in this repository.

📥 **Download model from Google Drive:**
https://drive.google.com/file/d/1HAUP__Glz4xxQpkXCJrDtwDdsec2Fw6k/view?usp=sharing

---

### 🔹 Setup Model Files

1. Download the file
2. Extract it
3. Place the folder in your project directory:

```
xpurl_saved/
```

---

### 🔹 Expected structure

```
xpurl_saved/
├── xpurl_best.pt
├── scaler.pkl
├── cfg.json
├── best_thresh.txt
├── feature_names.npy
├── metadata.json
├── shap_background.npy
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Barath-200/xpurl-explainable-phishing-detector.git
cd xpurl-explainable-phishing-detector
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the application

```bash
python app.py
```

---

### 4. Open in browser

```
http://localhost:5000
```

---

## 🔌 API Endpoints

### 🔹 Health Check

```
GET /api/health
```

---

### 🔹 Predict URL

```
POST /api/predict
```

```json
{
  "url": "http://example.com"
}
```

---

### 🔹 Batch Prediction

```
POST /api/predict_batch
```

---

### 🔹 Feature Extraction

```
POST /api/features
```

---

## 📊 Example Output

```json
{
  "url": "http://fake-login.com",
  "label": "PHISHING",
  "probability": 0.91,
  "risk": "HIGH",
  "shap_drivers": [
    {"feature": "brand_impersonation", "value": 0.4}
  ]
}
```

---

## 🧪 Demo Mode

If model files are not found, the system runs in **demo mode** using heuristic scoring.

---

## 🔒 Technologies Used

* Python
* Flask
* PyTorch
* Transformers (DistilBERT)
* Scikit-learn
* SHAP
* NumPy

---

## 📈 Future Enhancements

* 🔄 Continual learning (EWC live updates)
* 🌐 Browser extension
* ☁️ Cloud deployment
* 📊 Analytics dashboard

---

## 👨‍💻 Author

**Barath**


---

## ⭐ Tip

For full functionality, make sure the `xpurl_saved/` folder is correctly placed.

---
