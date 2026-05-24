# Marathi E-commerce Review NLP — Live Demo

Live web app that runs a complete Marathi NLP pipeline on user-supplied product reviews.

**Features**
- ✍️ **Analyze a Review** — paste any Marathi review and get sentiment (lexicon + TF-IDF + LR), Plutchik emotion, named entities, and POS tags live.
- 🤖 **Marathi Chatbot** — ask product questions in Marathi or English; retrieves the most similar review from a 560-row corpus.
- 📊 **Visualizations** — Devanagari wordcloud, top bigrams, sentiment-model comparison, confusion matrix, dataset breakdown.
- ℹ️ **About** — full methodology, dictionaries, and headline metrics.

**Headline numbers**
- 560-row Marathi product reviews corpus (5 categories)
- Sentiment classifier: **93% accuracy**, **0.91 macro-F1** (TF-IDF + Logistic Regression, 5-fold CV)
- Domain emotion lexicon (Plutchik tags + polarity + intensity)
- Retrieval-based chatbot over TF-IDF embeddings

---

## Run locally

```bash
git clone <your-repo-url>
cd marathi_nlp_streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically at `http://localhost:8501/`.

---

## Deploy to Streamlit Community Cloud (free, ~5 minutes)

See **DEPLOY.md** for the full step-by-step. Three-line summary:

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io/ → *New app* → connect your repo.
3. Click **Deploy**. You get a live `https://<your-app>.streamlit.app/` URL.

---

## File structure

```
marathi_nlp_streamlit/
├── app.py                  ← main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← this file
├── DEPLOY.md               ← step-by-step deployment guide
├── .gitignore
├── .streamlit/
│   └── config.toml         ← theme + server config
├── data/
│   ├── marathi_reviews_expanded.csv   (560 rows)
│   ├── marathi_reviews_sample.csv     (60-row seed)
│   ├── marathi_stopwords.txt
│   ├── ner_dict.json
│   ├── pos_dict.json
│   └── emotion_lexicon.json
└── fonts/
    └── NotoSansDevanagari-Regular.ttf  ← required for Marathi chart rendering
```

---

## Author

Pooja Kadam · MBA WE 5 — Natural Language Processing · May 2026

Course assignment: regional-language NLP on Marathi e-commerce product reviews.
