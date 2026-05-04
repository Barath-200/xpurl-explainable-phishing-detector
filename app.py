"""
XPURL Backend — Flask API
Explainable & Continually Adaptive Phishing URL Detection
"""

import os
import re
import math
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.preprocessing import StandardScaler
import tldextract
import shap

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

SAVE_DIR = os.environ.get("XPURL_MODEL_DIR", "./xpurl_saved")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG_DEFAULT = {
    "max_char_len"  : 64,
    "stat_feat_dim" : 25,
    "bert_hidden"   : 768,
    "fusion_dim"    : 256,
    "dropout"       : 0.3,
    "num_classes"   : 2,
}

BRAND_KEYWORDS = {
    "paypal","apple","amazon","microsoft","google","netflix","facebook",
    "instagram","chase","wellsfargo","bankofamerica","citibank","usps",
    "fedex","irs","ebay","steam","dropbox","linkedin","twitter",
}

SUS_TLDS = {
    "ru","cn","tk","ml","ga","cf","gq","xyz","top","club","info",
    "pw","cc","biz","online","site","website","live","stream",
}

FEATURE_NAMES_DEFAULT = [
    "url_length","netloc_length","dot_count","dash_count","underscore_count",
    "slash_count","question_count","equals_count","at_count","percent_count",
    "digit_ratio","upper_ratio","url_entropy","domain_entropy","path_entropy",
    "subdomain_depth","path_depth","query_length","has_ip","brand_impersonation",
    "sus_char_ratio","sus_tld","has_punycode","has_sus_keywords","complexity_score",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Model Definition  (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────

class XPURLModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.stat_proj = nn.Sequential(
            nn.Linear(cfg["stat_feat_dim"], 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(64, 128), nn.GELU(),
        )
        fused_dim = cfg["bert_hidden"] + 128
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim, 2), nn.Softmax(dim=-1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, cfg["fusion_dim"]), nn.LayerNorm(cfg["fusion_dim"]),
            nn.GELU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["fusion_dim"], 64), nn.GELU(),
            nn.Dropout(cfg["dropout"] * 0.5),
            nn.Linear(64, cfg["num_classes"]),
        )

    def forward(self, input_ids, attention_mask, stat_feats):
        bert_out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed  = bert_out.last_hidden_state[:, 0, :]
        stat_embed = self.stat_proj(stat_feats)
        fused      = torch.cat([cls_embed, stat_embed], dim=-1)
        return self.classifier(fused)


class StatOnlyWrapper:
    """Wraps statistical head for SHAP explainability."""
    def __init__(self, model, device):
        self.model  = model
        self.device = device
        dummy_ids  = torch.zeros(1, CFG_DEFAULT["max_char_len"], dtype=torch.long).to(device)
        dummy_mask = torch.ones(1, CFG_DEFAULT["max_char_len"], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model.bert(input_ids=dummy_ids, attention_mask=dummy_mask)
            self.mean_cls = out.last_hidden_state[:, 0, :].detach()
        model.eval()

    def predict_proba(self, stat_feats_np):
        results = []
        for i in range(0, len(stat_feats_np), 32):
            chunk  = stat_feats_np[i:i+32]
            stat_t = torch.tensor(chunk, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                stat_embed  = self.model.stat_proj(stat_t)
                bert_expand = self.mean_cls.expand(stat_t.size(0), -1)
                fused       = torch.cat([bert_expand, stat_embed], dim=-1)
                probs       = F.softmax(self.model.classifier(fused), dim=-1).cpu().numpy()
            results.append(probs)
        return np.vstack(results)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Extraction  (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = defaultdict(int)
    for c in s:
        freq[c] += 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def extract_features(url: str) -> np.ndarray:
    url = str(url).strip()
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        return np.zeros(25, dtype=np.float32)

    ext    = tldextract.extract(url)
    netloc = parsed.netloc or ""
    path   = parsed.path   or ""
    query  = parsed.query  or ""
    full   = url

    sus_chars     = set("@#%~|\\^`<>")
    brand_in_url  = any(b in full.lower() for b in BRAND_KEYWORDS)
    brand_sus_tld = 1 if (brand_in_url and ext.suffix.lower() in SUS_TLDS) else 0

    return np.array([
        len(full),
        len(netloc),
        full.count("."),
        full.count("-"),
        full.count("_"),
        full.count("/"),
        full.count("?"),
        full.count("="),
        full.count("@"),
        full.count("%"),
        sum(c.isdigit() for c in full) / max(len(full), 1),
        sum(c.isupper() for c in full) / max(len(full), 1),
        shannon_entropy(full),
        shannon_entropy(ext.domain),
        shannon_entropy(path),
        netloc.count(".") - 1,
        len(path.split("/")),
        len(query),
        1 if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", netloc) else 0,
        brand_sus_tld,
        sum(c in sus_chars for c in full) / max(len(full), 1),
        1 if ext.suffix.lower() in SUS_TLDS else 0,
        1 if "xn--" in netloc.lower() else 0,
        1 if re.search(
            r"(login|verify|secure|update|account|banking|"
            r"confirm|password|signin|webscr|suspended)",
            full, re.I,
        ) else 0,
        (full.count(".") + full.count("-") + full.count("/") +
         full.count("%") + shannon_entropy(full)) / 10.0,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────────────

model         = None
tokenizer     = None
scaler        = None
explainer     = None
BEST_THRESH   = 0.5
CFG           = CFG_DEFAULT.copy()
FEATURE_NAMES = FEATURE_NAMES_DEFAULT[:]
model_meta    = {}


def char_tokenize(url: str, max_len: int = 64):
    char_seq = " ".join(list(url.lower()[:max_len * 2]))
    enc = tokenizer(
        char_seq, max_length=max_len,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def load_model():
    global model, tokenizer, scaler, explainer
    global BEST_THRESH, CFG, FEATURE_NAMES, model_meta

    if not os.path.isdir(SAVE_DIR):
        print(f"[XPURL] Model directory '{SAVE_DIR}' not found. Running in demo mode.")
        return False

    try:
        with open(f"{SAVE_DIR}/cfg.json") as f:
            CFG = json.load(f)
        with open(f"{SAVE_DIR}/best_thresh.txt") as f:
            BEST_THRESH = float(f.read().strip())
        with open(f"{SAVE_DIR}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        FEATURE_NAMES_ARR = np.load(f"{SAVE_DIR}/feature_names.npy", allow_pickle=True)
        FEATURE_NAMES     = FEATURE_NAMES_ARR.tolist()

        with open(f"{SAVE_DIR}/metadata.json") as f:
            model_meta = json.load(f)

        print(f"[XPURL] Config loaded | threshold={BEST_THRESH:.3f}")

        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased", do_lower_case=True
        )

        model = XPURLModel(CFG).to(DEVICE)
        weights_path = f"{SAVE_DIR}/xpurl_ewc.pt"
        if not os.path.exists(weights_path):
            weights_path = f"{SAVE_DIR}/xpurl.pt"
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        print("[XPURL] Model weights loaded.")

        background_path = f"{SAVE_DIR}/shap_background.npy"
        if os.path.exists(background_path):
            background_feats = np.load(background_path)
            wrapper          = StatOnlyWrapper(model, DEVICE)
            explainer        = shap.KernelExplainer(wrapper.predict_proba, background_feats)
            print(f"[XPURL] SHAP explainer ready | background: {background_feats.shape}")

        return True

    except Exception as exc:
        print(f"[XPURL] Model load error: {exc}")
        return False


def _demo_predict(url: str) -> dict:
    """Deterministic heuristic prediction used when no model is loaded."""
    raw  = extract_features(url)
    feat = dict(zip(FEATURE_NAMES_DEFAULT, raw.tolist()))

    score = 0.0
    score += min(raw[0] / 200.0, 0.3)          # url length
    score += raw[19] * 0.4                       # brand_impersonation
    score += raw[21] * 0.15                      # sus_tld
    score += raw[23] * 0.1                       # sus_keywords
    score += raw[18] * 0.15                      # has_ip
    score += raw[12] / 30.0 * 0.1               # entropy
    prob  = float(np.clip(score, 0.0, 0.99))

    pred  = int(prob >= 0.5)
    label = "PHISHING" if pred else "LEGITIMATE"
    risk  = "HIGH" if prob > 0.8 else ("MEDIUM" if prob > 0.5 else "LOW")

    shap_drivers = [
        {"feature": "brand_impersonation", "value": float(raw[19]) * 0.4},
        {"feature": "url_length",          "value": float(raw[0]) / 200.0 * 0.3},
        {"feature": "sus_tld",             "value": float(raw[21]) * 0.15},
        {"feature": "has_ip",              "value": float(raw[18]) * 0.15},
        {"feature": "sus_keywords",        "value": float(raw[23]) * 0.1},
    ]

    return {
        "url"         : url,
        "label"       : label,
        "probability" : round(prob, 4),
        "risk"        : risk,
        "threshold"   : 0.5,
        "shap_drivers": shap_drivers,
        "features"    : {k: round(v, 4) for k, v in feat.items()},
        "mode"        : "demo",
    }


def predict_url(url: str) -> dict:
    if model is None or tokenizer is None or scaler is None:
        return _demo_predict(url)

    url = url.strip()
    if not url:
        return {"error": "Empty URL"}

    raw_feat = extract_features(url).reshape(1, -1)
    scl_feat = scaler.transform(raw_feat)
    stat_t   = torch.tensor(scl_feat, dtype=torch.float32).to(DEVICE)

    ids, mask = char_tokenize(url, CFG.get("max_char_len", 64))
    ids   = ids.unsqueeze(0).to(DEVICE)
    mask  = mask.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(ids, mask, stat_t)
        probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    phish_prob = float(probs[1])
    pred       = int(phish_prob >= BEST_THRESH)
    label      = "PHISHING" if pred == 1 else "LEGITIMATE"
    risk       = "HIGH" if phish_prob > 0.8 else ("MEDIUM" if phish_prob > 0.5 else "LOW")

    shap_drivers = []
    if explainer is not None:
        try:
            sv       = explainer.shap_values(scl_feat, nsamples=50)
            sv_arr   = np.array(sv)
            row      = sv_arr[0, :, 1] if sv_arr.ndim == 3 else (
                np.array(sv[1]).flatten() if isinstance(sv, list) else sv_arr.flatten()
            )
            top5_idx = np.argsort(np.abs(row))[::-1][:5]
            shap_drivers = [
                {"feature": FEATURE_NAMES[int(i)], "value": round(float(row[int(i)]), 6)}
                for i in top5_idx
            ]
        except Exception:
            pass

    return {
        "url"         : url,
        "label"       : label,
        "probability" : round(phish_prob, 4),
        "risk"        : risk,
        "threshold"   : round(BEST_THRESH, 4),
        "shap_drivers": shap_drivers,
        "features"    : {
            k: round(float(v), 4) for k, v in zip(FEATURE_NAMES, raw_feat[0])
        },
        "mode"        : "model",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Flask Application
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return send_file(html_path)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status"   : "ok",
        "model"    : "loaded" if model is not None else "demo",
        "device"   : str(DEVICE),
        "threshold": round(BEST_THRESH, 4),
        "metadata" : model_meta,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    url  = (data or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    return jsonify(predict_url(url))


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    urls = (data or {}).get("urls", [])
    if not urls or not isinstance(urls, list):
        return jsonify({"error": "Missing 'urls' list"}), 400
    if len(urls) > 50:
        return jsonify({"error": "Max 50 URLs per batch"}), 400
    results = [predict_url(u) for u in urls]
    return jsonify({"results": results, "count": len(results)})


@app.route("/api/features", methods=["POST"])
def features_only():
    data = request.get_json(force=True)
    url  = (data or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    raw = extract_features(url)
    return jsonify({
        "url"     : url,
        "features": {k: round(float(v), 4) for k, v in zip(FEATURE_NAMES_DEFAULT, raw)},
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[XPURL] Starting server...")
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
