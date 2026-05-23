# Twitter Sentiment Analysis

A full end-to-end machine learning pipeline that classifies tweets into **Positive**, **Negative**, and **Neutral** sentiments — with a fine-grained 5-class extension (Strongly Positive → Strongly Negative). Built with BERT embeddings and three trained classifiers, deployed via an interactive Streamlit dashboard.

---

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| SVM (best) | **63.26%** | **63.15%** |
| Logistic Regression | 61.39% | 61.38% |
| Naive Bayes | 53.28% | 53.16% |

> SVM with BERT embeddings achieved the best performance. All models were evaluated on a held-out test set of 4,815 tweets.

---

## Project Overview

Social media platforms generate millions of opinions daily. Manual analysis is impossible at scale. This project automates sentiment classification of tweets to enable faster insight extraction for researchers, brands, and analysts.

**Key design decisions:**
- Used **BERT embeddings** (`bert-base-uncased`) instead of TF-IDF for richer semantic representations
- Applied **smart imputation** (zero rows dropped) rather than removing missing data
- Preserved **negation words** (not, never, don't, etc.) during stopword removal to improve accuracy
- Built a **5-class fine-grained label** (strongly positive/positive/neutral/negative/strongly negative) using selected_text confidence ratio
- Deployed a **Streamlit frontend** with live prediction, EDA visualizations, and model comparison

---

## Pipeline

```
Raw Data → EDA (Before) → Smart Preprocessing → EDA (After)
       → BERT Embeddings → Train 3 Models → Evaluate → Streamlit App
```

1. **Data loading** — 6,160 training rows, 4,815 test rows (10 columns each)
2. **EDA before preprocessing** — sentiment distribution, word clouds, demographics, text length
3. **Smart preprocessing** — URL/mention removal, lemmatization, negation-aware stopword filtering
4. **5-class label engineering** — confidence ratio from selected_text span
5. **BERT embeddings** — sentence-level representations via HuggingFace Transformers
6. **Model training** — Logistic Regression, Gaussian Naive Bayes, SVM (scikit-learn)
7. **Evaluation** — accuracy, F1, confusion matrix, cross-validation
8. **Streamlit deployment** — live tweet input, model selector, visual dashboard

---

## Dataset

| Split | Rows | Columns |
|---|---|---|
| Train | 6,160 | 10 |
| Test | 4,815 | 10 |

**Columns:** `textID`, `text`, `selected_text`, `sentiment`, `Time of Tweet`, `Age of User`, `Country`, `Population-2020`, `Land Area (Km²)`, `Density (P/Km²)`

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / Modeling | scikit-learn (LR, GaussianNB, SVC) |
| NLP | NLTK, HuggingFace Transformers (BERT) |
| Embeddings | `bert-base-uncased` via PyTorch |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn, plotly, WordCloud |
| Frontend | Streamlit |
| Persistence | joblib (.pkl models) |

---

## Project Structure

```
twitter-sentiment-analysis/
├── app.py                          # Streamlit frontend (live prediction + dashboard)
├── requirements.txt                # All dependencies
├── notebooks/
│   └── sentiment_analysis_pipeline.ipynb   # Full ML pipeline (EDA → BERT → Train → Eval)
├── models/
│   ├── model_meta.json             # Accuracy & F1 scores for all 3 models
│   └── README_models.md            # How to regenerate .pkl files
├── data/
│   └── README_data.md              # Dataset description and column guide
└── results/                        # EDA plots and confusion matrices (generated on run)
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full ML pipeline
```bash
jupyter notebook notebooks/sentiment_analysis_pipeline.ipynb
```
This trains all three models and saves `lr_model.pkl`, `nb_model.pkl`, `svm_model.pkl`, and `label_encoder.pkl` to the project root.

### 3. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

---

## Key Findings

- **SVM outperformed** Logistic Regression and Naive Bayes when combined with BERT embeddings, achieving 63.26% accuracy on a 3-class problem
- **Neutral tweets** were the most difficult to classify correctly due to ambiguous and context-dependent language
- **Negation preservation** (keeping words like "not", "never", "don't" during preprocessing) measurably improved sentiment boundary detection
- **BERT embeddings** provided richer semantic context compared to bag-of-words approaches, especially for sarcastic or ironic tweets
- **Demographics analysis** revealed that tweet sentiment varies by time-of-day and user age group

---

## Authors

| Name | Student ID |
|---|---|
| Minahil Ashraf | AI053 |
| [Team Member 2] | AI034 |
| [Team Member 3] | AI003 |

BS Artificial Intelligence — PAF-IAST, Mang, Haripur, Pakistan

---

## Contact

**Minahil Ashraf** — minahillll29@gmail.com  
[LinkedIn](#) | PAF-IAST, Pakistan
