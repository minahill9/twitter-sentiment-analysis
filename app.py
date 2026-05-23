"""
Twitter Sentiment Analysis — Streamlit Frontend
Run with:  streamlit run app.py
Requires:  lr_model.pkl, nb_model.pkl, svm_model.pkl,
           label_encoder.pkl, model_meta.json, train_for_frontend.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib, json, re, os, time
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import torch
from transformers import AutoTokenizer, AutoModel

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Colour scheme ────────────────────────────────────────────────────────────
COLORS5 = {
    "strongly positive": "#1a9641",
    "positive":          "#a6d96a",
    "neutral":           "#3498db",
    "negative":          "#fdae61",
    "strongly negative": "#d7191c",
}
EMOJIS = {
    "strongly positive": "🌟",
    "positive":          "😊",
    "neutral":           "😐",
    "negative":          "😞",
    "strongly negative": "😡",
}

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    /* Cards */
    .card {
        background: #1e2130;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 18px;
        border: 1px solid #2e3350;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    /* Sentiment badge */
    .badge {
        display: inline-block;
        padding: 8px 22px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 6px 4px;
    }
    /* Accuracy bar wrapper */
    .acc-bar-wrap { background:#2c2f3f; border-radius:8px; height:22px; margin:4px 0; overflow:hidden; }
    .acc-bar-fill { height:100%; border-radius:8px; transition:width 1s; }
    /* Model card header */
    .model-title { font-size:17px; font-weight:700; margin-bottom:4px; }
    .metric-num  { font-size:28px; font-weight:800; }
    /* Tab bar tweak */
    .stTabs [data-baseweb="tab-list"] { gap:12px; }
    .stTabs [data-baseweb="tab"]      { border-radius:8px 8px 0 0; padding:8px 20px; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background:#151824; }
    /* Divider */
    hr { border-color: #2e3350; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCE LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading BERT model…")
def load_bert():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("punkt",     quiet=True)
    model_name = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    return tokenizer, model, device


@st.cache_resource(show_spinner="Loading classifiers…")
def load_classifiers():
    lr  = joblib.load("lr_model.pkl")
    nb  = joblib.load("nb_model.pkl")
    svm = joblib.load("svm_model.pkl")
    le  = joblib.load("label_encoder.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return lr, nb, svm, le, meta


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    return pd.read_csv("train_for_frontend.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT PROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_nlp_tools():
    stop = set(stopwords.words("english"))
    neg  = {"not","no","nor","never","nobody","nothing","neither","nowhere","none",
            "cannot","cant","dont","wont","isnt","arent","wasnt","werent",
            "hasnt","havent","hadnt","doesnt","didnt","wouldnt","shouldnt","couldnt"}
    stop -= neg
    lem  = WordNetLemmatizer()
    return stop, lem


def clean_text(txt):
    stop, lem = get_nlp_tools()
    txt = str(txt).lower()
    txt = re.sub(r"http\S+|www\S+",  " ", txt)
    txt = re.sub(r"@\w+",            " ", txt)
    txt = re.sub(r"#(\w+)",       r"\1", txt)
    txt = re.sub(r"[^\x00-\x7F]+",  " ", txt)
    txt = re.sub(r"[^\w\s]",        " ", txt)
    txt = re.sub(r"\d+",            " ", txt)
    tokens = [lem.lemmatize(t) for t in txt.split()
              if t not in stop and len(t) > 1]
    return " ".join(tokens)


def embed_text(text, tokenizer, bert_model, device):
    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = bert_model(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy()


def base_to_fine(base_label: str, confidence: float) -> str:
    """Map 3-class prediction + confidence → 5-class label."""
    if base_label == "positive":
        return "strongly positive" if confidence >= 0.70 else "positive"
    elif base_label == "negative":
        return "strongly negative" if confidence >= 0.70 else "negative"
    return "neutral"


def predict_tweet(text, tokenizer, bert_model, device, lr, nb, svm, le):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return None
    emb = embed_text(cleaned, tokenizer, bert_model, device)
    results = {}
    for name, model in [("Logistic Regression", lr), ("Naive Bayes", nb), ("SVM", svm)]:
        base  = le.inverse_transform(model.predict(emb))[0]
        proba = model.predict_proba(emb)[0]
        conf  = float(max(proba))
        fine  = base_to_fine(base, conf)
        results[name] = {
            "fine":       fine,
            "base":       base,
            "confidence": conf,
            "proba":      dict(zip(le.classes_, proba.round(4))),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    ax.tick_params(colors="#cccccc"); ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc"); ax.title.set_color("#ffffff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3f5c")
    return fig, ax


def dark_fig_multi(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor("#1e2130")
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3f5c")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#ffffff")
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════

tokenizer, bert_model, device = load_bert()
lr, nb, svm, le, meta         = load_classifiers()
df                             = load_data()

# Pre-compute 5-class counts from the dataset
ordered5 = ["strongly positive", "positive", "neutral", "negative", "strongly negative"]
five_counts = df["fine_label"].value_counts().reindex(ordered5).fillna(0).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🐦 Twitter Sentiment Analyzer")
    st.markdown("---")

    st.markdown("### 📊 Model Accuracy")
    model_colors = {"Logistic Regression": "#3498db", "Naive Bayes": "#e67e22", "SVM": "#9b59b6"}
    for m_name, m_col in model_colors.items():
        acc_pct = meta[m_name]["accuracy"] * 100
        f1_pct  = meta[m_name]["f1"]       * 100
        st.markdown(f"""
        <div class='card' style='padding:14px 18px;'>
            <div class='model-title' style='color:{m_col};'>{m_name}</div>
            <div>Accuracy: <b>{acc_pct:.1f}%</b></div>
            <div class='acc-bar-wrap'>
                <div class='acc-bar-fill' style='width:{acc_pct:.1f}%;background:{m_col};'></div>
            </div>
            <div style='font-size:13px;color:#aaa;'>Weighted F1: {f1_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Dataset Stats")
    st.metric("Training samples", f"{len(df):,}")
    st.metric("Classes", "5 (fine-grained)")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    - **Embeddings:** DistilBERT (CLS token)
    - **Models:** LR · Naive Bayes · SVM
    - **Labels:** 5-class fine-grained
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🐦 Twitter Sentiment Analysis")
tab_predict, tab_dataset, tab_models, tab_why = st.tabs(
    ["🔍 Predict Tweet", "📊 Dataset Overview", "📈 Model Performance", "💡 Why This Sentiment?"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("### Enter a tweet to analyse its sentiment across all three models")

    tweet_input = st.text_area(
        "Tweet text",
        placeholder="e.g. I absolutely love this product! Best purchase ever 😍",
        height=110,
        key="tweet_input"
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    analyze_btn = col_btn1.button("🔍 Analyse", type="primary", use_container_width=True)
    clear_btn   = col_btn2.button("🗑️ Clear",   use_container_width=True)

    if clear_btn:
        st.rerun()

    # Quick example tweets
    st.markdown("**Try an example:**")
    ex_cols = st.columns(4)
    examples = [
        "I absolutely love this! Best day ever!!!",
        "I hate everything about this, it's terrible.",
        "Just woke up and heading to work.",
        "Not happy at all. This is so wrong and painful.",
    ]
    for col, ex in zip(ex_cols, examples):
        if col.button(ex[:30] + "…", key=f"ex_{ex[:10]}", use_container_width=True):
            tweet_input = ex
            analyze_btn = True

    if analyze_btn and tweet_input.strip():
        with st.spinner("Embedding with BERT and predicting…"):
            results = predict_tweet(
                tweet_input, tokenizer, bert_model, device, lr, nb, svm, le
            )

        if results is None:
            st.warning("⚠️ Tweet is empty after cleaning. Please enter more meaningful text.")
        else:
            st.markdown("---")
            st.markdown("#### 🎯 Predictions by Model")

            cols = st.columns(3)
            m_cols = {"Logistic Regression": "#3498db", "Naive Bayes": "#e67e22", "SVM": "#9b59b6"}

            for col, (m_name, m_col) in zip(cols, m_cols.items()):
                res  = results[m_name]
                fine = res["fine"]
                emoji = EMOJIS[fine]
                badge_col = COLORS5[fine]
                conf_pct  = res["confidence"] * 100

                with col:
                    st.markdown(f"""
                    <div class='card'>
                        <div style='font-size:16px;font-weight:700;color:{m_col};margin-bottom:10px;'>{m_name}</div>
                        <div style='text-align:center;margin:10px 0;'>
                            <span class='badge' style='background:{badge_col};color:white;font-size:20px;'>
                                {emoji} {fine.title()}
                            </span>
                        </div>
                        <div style='font-size:13px;color:#aaa;margin-top:8px;'>
                            Base class: <b>{res['base'].capitalize()}</b> &nbsp;|&nbsp;
                            Confidence: <b>{conf_pct:.1f}%</b>
                        </div>
                        <div class='acc-bar-wrap' style='margin-top:8px;'>
                            <div class='acc-bar-fill' style='width:{conf_pct:.1f}%;background:{badge_col};'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Probability breakdown chart
            st.markdown("#### 📊 Probability Breakdown per Model")
            fig, axes = dark_fig_multi(1, 3, w=14, h=4)

            for ax, (m_name, m_col) in zip(axes, m_cols.items()):
                proba = results[m_name]["proba"]
                classes = list(proba.keys())
                vals    = [proba[c]*100 for c in classes]
                bar_cols = [COLORS5.get(c, "#888") if c in COLORS5 else
                            {"positive":"#a6d96a","negative":"#fdae61","neutral":"#3498db"}.get(c,"#888")
                            for c in classes]
                bars = ax.bar(classes, vals, color=bar_cols, edgecolor="white", linewidth=1.2)
                ax.set_title(m_name, fontweight="bold", color=m_col, fontsize=13)
                ax.set_ylabel("Probability (%)")
                ax.set_ylim(0, 110)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                            f"{v:.1f}%", ha="center", fontsize=10, color="#ddd", fontweight="bold")
                ax.tick_params(colors="#cccccc")

            fig.suptitle("Class Probability Distribution", color="white", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Model agreement
            fine_labels = [results[m]["fine"] for m in m_cols]
            agreed = len(set(fine_labels)) == 1
            if agreed:
                st.success(f"✅ All three models **agree**: the tweet is **{fine_labels[0].title()}** {EMOJIS[fine_labels[0]]}")
            else:
                st.info(f"ℹ️ Models differ: LR→ {fine_labels[0]} | NB→ {fine_labels[1]} | SVM→ {fine_labels[2]}")

    elif analyze_btn:
        st.warning("Please enter a tweet first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_dataset:
    st.markdown("### 📊 Dataset Sentiment Distribution")
    st.markdown("Colour-coded breakdown of all **5 fine-grained sentiment classes** in the training set.")

    col_left, col_right = st.columns(2)

    # --- Horizontal stacked bar (absolute) ---
    with col_left:
        fig, ax = dark_fig(7, 4)
        left_offset = 0
        total = five_counts.sum()
        for label in ordered5:
            val = five_counts[label]
            ax.barh(["Dataset"], [val], left=left_offset,
                    color=COLORS5[label], label=label, edgecolor="white", linewidth=0.8)
            if val > 30:
                ax.text(left_offset + val/2, 0, f"{val}\n{val/total*100:.1f}%",
                        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            left_offset += val
        ax.set_title("Tweet Count by Sentiment (Stacked)", color="white", fontweight="bold")
        ax.set_xlabel("Number of Tweets")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3,
                  facecolor="#1e2130", labelcolor="white", edgecolor="#3a3f5c")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # --- Donut chart ---
    with col_right:
        fig, ax = dark_fig(6, 4)
        wedges, texts, autotexts = ax.pie(
            five_counts.values, labels=None,
            autopct="%1.1f%%", colors=[COLORS5[l] for l in ordered5],
            startangle=140, wedgeprops={"edgecolor":"white","linewidth":2},
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(10); at.set_fontweight("bold")
        # Draw hole for donut
        centre = plt.Circle((0,0), 0.5, color="#1e2130")
        ax.add_artist(centre)
        ax.text(0, 0, f"{total:,}\ntweets", ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        legend_patches = [mpatches.Patch(color=COLORS5[l], label=l.title()) for l in ordered5]
        ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.3),
                  ncol=2, facecolor="#1e2130", labelcolor="white", edgecolor="#3a3f5c")
        ax.set_title("Sentiment Proportion (Donut)", color="white", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # --- Bar chart per sentiment class with tooltip counts ---
    st.markdown("#### Fine-grained Class Counts")
    fig, ax = dark_fig(11, 5)
    bars = ax.bar(ordered5, five_counts.values,
                  color=[COLORS5[l] for l in ordered5],
                  edgecolor="white", linewidth=1.2, width=0.55)
    for bar, v in zip(bars, five_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15,
                str(int(v)), ha="center", fontsize=12, fontweight="bold", color="#ddd")
    ax.set_title("Number of Tweets per Fine-grained Sentiment Class",
                 color="white", fontweight="bold", fontsize=14)
    ax.set_xlabel("Sentiment Class"); ax.set_ylabel("Count")
    ax.set_xticklabels([l.title() for l in ordered5], rotation=12)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Metrics row
    m_cols = st.columns(5)
    for col, label in zip(m_cols, ordered5):
        col.metric(label.title(), int(five_counts[label]),
                   delta=f"{five_counts[label]/total*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_models:
    st.markdown("### 📈 Model Accuracy Comparison")

    model_names_list = list(meta.keys())
    accs = [meta[m]["accuracy"]*100 for m in model_names_list]
    f1s  = [meta[m]["f1"]*100       for m in model_names_list]
    mcols_list = ["#3498db", "#e67e22", "#9b59b6"]

    # Big accuracy metric cards
    mc = st.columns(3)
    for col, name, acc, f1, mc_col in zip(mc, model_names_list, accs, f1s, mcols_list):
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center;'>
                <div style='font-size:15px;font-weight:700;color:{mc_col};'>{name}</div>
                <div class='metric-num' style='color:{mc_col};margin:10px 0;'>{acc:.1f}%</div>
                <div style='font-size:13px;color:#aaa;'>Accuracy</div>
                <div class='acc-bar-wrap' style='margin:10px 0;'>
                    <div class='acc-bar-fill' style='width:{acc:.1f}%;background:{mc_col};'></div>
                </div>
                <div style='font-size:13px;color:#bbb;'>Weighted F1: <b>{f1:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Grouped bar chart
    fig, ax = dark_fig(11, 6)
    x = np.arange(len(model_names_list))
    w = 0.32
    b1 = ax.bar(x-w/2, accs, w, label="Accuracy (%)",    color=mcols_list, alpha=0.9)
    b2 = ax.bar(x+w/2, f1s,  w, label="Weighted F1 (%)", color=mcols_list, alpha=0.45, hatch="///")
    ax.set_xticks(x); ax.set_xticklabels(model_names_list, fontsize=12, color="#ccc")
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 108)
    ax.set_title("Model Comparison — Accuracy & Weighted F1", color="white", fontweight="bold", fontsize=14)
    ax.legend(facecolor="#1e2130", labelcolor="white", edgecolor="#3a3f5c")
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10, color="#ddd", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Accuracy table
    st.markdown("#### 📋 Summary Table")
    summary = pd.DataFrame({
        "Model":    model_names_list,
        "Accuracy": [f"{a:.2f}%" for a in accs],
        "F1 Score": [f"{f:.2f}%" for f in f1s],
        "Embeddings": ["BERT (DistilBERT CLS)"] * 3,
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info("""
    **Notes:**
    - All models use **DistilBERT CLS-token embeddings** as features.
    - Accuracy reported on 20% stratified validation split.
    - SVM typically achieves the highest accuracy on BERT embeddings for this task.
    - Naive Bayes is fastest but less precise with dense embeddings.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — WHY THIS SENTIMENT (keyword insight)
# ─────────────────────────────────────────────────────────────────────────────
with tab_why:
    st.markdown("### 💡 Why are tweets Positive or Negative?")
    st.markdown("Explore the trigger keywords that drive each sentiment class in the dataset.")

    POS_KEYS = ["love","great","good","happy","thank","awesome","nice","best",
                "wonderful","excited","amazing","glad","fun","enjoy","yay",
                "beautiful","fantastic","perfect","brilliant","lucky"]
    NEG_KEYS = ["hate","sad","bad","terrible","worst","awful","miss","cry",
                "annoyed","angry","fail","sick","wrong","hurt","pain",
                "bored","stupid","disgusting","horrible","lonely"]

    pos_txt = " ".join(df[df["sentiment"]=="positive"]["clean_text"].dropna())
    neg_txt = " ".join(df[df["sentiment"]=="negative"]["clean_text"].dropna())

    pos_freq = pd.Series({k: pos_txt.split().count(k) for k in POS_KEYS}).sort_values(ascending=False)
    neg_freq = pd.Series({k: neg_txt.split().count(k) for k in NEG_KEYS}).sort_values(ascending=False)

    fig, axes = dark_fig_multi(1, 2, w=15, h=6)

    axes[0].barh(pos_freq.index[::-1], pos_freq.values[::-1], color="#2ecc71")
    axes[0].set_title("Positive Trigger Words", color="#2ecc71", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Occurrence count in positive tweets")
    for i, v in enumerate(pos_freq.values[::-1]):
        axes[0].text(v+1, i, str(int(v)), va="center", color="#ddd", fontsize=10)

    axes[1].barh(neg_freq.index[::-1], neg_freq.values[::-1], color="#e74c3c")
    axes[1].set_title("Negative Trigger Words", color="#e74c3c", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Occurrence count in negative tweets")
    for i, v in enumerate(neg_freq.values[::-1]):
        axes[1].text(v+1, i, str(int(v)), va="center", color="#ddd", fontsize=10)

    fig.suptitle("Keyword Frequency Analysis — WHY are tweets positive/negative?",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Sentiment keyword checker
    st.markdown("---")
    st.markdown("#### 🔎 Check a word's sentiment signal")
    word_check = st.text_input("Enter a word:", placeholder="e.g. love")
    if word_check:
        w = word_check.lower().strip()
        pos_c = pos_txt.split().count(w)
        neg_c = neg_txt.split().count(w)
        neu_c = " ".join(df[df["sentiment"]=="neutral"]["clean_text"].dropna()).split().count(w)
        mc2 = st.columns(3)
        mc2[0].metric("In Positive tweets", pos_c)
        mc2[1].metric("In Negative tweets", neg_c)
        mc2[2].metric("In Neutral tweets",  neu_c)
        if pos_c > neg_c and pos_c > neu_c:
            st.success(f"✅ **'{w}'** is predominantly a **positive** indicator in this dataset.")
        elif neg_c > pos_c and neg_c > neu_c:
            st.error(f"⚠️ **'{w}'** is predominantly a **negative** indicator in this dataset.")
        else:
            st.info(f"ℹ️ **'{w}'** appears frequently across multiple classes — likely neutral.")

    # Top-10 words per class heat
    st.markdown("---")
    st.markdown("#### 🌡️ Top-15 Words Frequency Heatmap")

    top_words = {}
    for s in ["positive", "negative", "neutral"]:
        txt = " ".join(df[df["sentiment"]==s]["clean_text"].dropna())
        top_words[s] = dict(Counter(txt.split()).most_common(15))

    all_words = list({w for wds in top_words.values() for w in wds})
    heatmap_df = pd.DataFrame(
        {s: [top_words[s].get(w, 0) for w in all_words] for s in ["positive","negative","neutral"]},
        index=all_words
    ).sort_values("positive", ascending=False).head(20)

    fig, ax = dark_fig(12, 7)
    sns.heatmap(heatmap_df, cmap="YlOrRd", ax=ax, annot=True, fmt="d",
                linewidths=0.5, linecolor="#1e2130",
                cbar_kws={"label":"Frequency"})
    ax.set_title("Word Frequency Heatmap across Sentiment Classes", color="white",
                 fontweight="bold", fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), color="#ccc")
    ax.set_yticklabels(ax.get_yticklabels(), color="#ccc", rotation=0)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
