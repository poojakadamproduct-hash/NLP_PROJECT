"""
Marathi E-commerce Review NLP — Streamlit App
==============================================
 
End-to-end demo of a regional-language NLP pipeline. Features:
  - Live sentiment + emotion + NER + POS analysis on any Marathi input
  - Retrieval-based Marathi chatbot
  - Dataset visualizations (wordcloud, top n-grams)
  - Project overview and methodology
 
Author: Pooja Kadam · Course: MBA WE 5 — NLP · 2026
"""
import os
import io
import re
import json
import unicodedata
from collections import Counter
from itertools import islice
from pathlib import Path
 
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
 
# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title='Marathi NLP — E-commerce Reviews',
    page_icon='📱',
    layout='wide',
    initial_sidebar_state='expanded',
)
 
# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS — visible loader instead of default gray-out on rerun
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* Animated progress bar at the top of the page during any rerun */
    [data-testid="stStatusWidget"] {
        position: fixed !important;
        top: 0 !important; left: 0 !important; right: 0 !important;
        width: 100vw !important; height: 3px !important;
        background: transparent !important;
        z-index: 999999 !important;
        padding: 0 !important; margin: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stStatusWidget"] > div {
        display: block !important;
        height: 3px !important;
        width: 100% !important;
        background: linear-gradient(90deg,
            rgba(31, 58, 95, 0)   0%,
            rgba(31, 58, 95, 0.9) 50%,
            rgba(31, 58, 95, 0)   100%) !important;
        background-size: 50% 100% !important;
        animation: nlp-progress 1.1s linear infinite !important;
        border-radius: 0 !important;
        color: transparent !important;
    }
    [data-testid="stStatusWidget"] svg,
    [data-testid="stStatusWidget"] button,
    [data-testid="stStatusWidget"] span,
    [data-testid="stStatusWidget"] p { display: none !important; }
 
    @keyframes nlp-progress {
        0%   { background-position: -100% 0; }
        100% { background-position:  200% 0; }
    }
 
    /* Smoother fade on page reruns instead of harsh gray-out */
    [data-testid="stAppViewContainer"] {
        transition: opacity 180ms ease-out;
    }
 
    /* Nicer spinner styling — bigger, centered */
    [data-testid="stSpinner"] > div {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        padding: 24px; gap: 12px;
    }
    [data-testid="stSpinner"] > div > div {
        border-color: #1F3A5F !important;
        border-top-color: transparent !important;
        width: 48px !important; height: 48px !important;
        border-width: 4px !important;
    }
    [data-testid="stSpinner"] > div > div + div {
        font-size: 1.1rem; color: #1F3A5F; font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ═══════════════════════════════════════════════════════════════════════
# PATHS & FONT REGISTRATION
# ═══════════════════════════════════════════════════════════════════════
BASE  = Path(__file__).parent
DATA  = BASE / 'data'
FONTS = BASE / 'fonts'
 
FONT_PATH = FONTS / 'NotoSansDevanagari-Regular.ttf'
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    DEVFONT = fm.FontProperties(fname=str(FONT_PATH))
    plt.rcParams['font.family'] = DEVFONT.get_name()
else:
    DEVFONT = None
 
# ═══════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA / 'marathi_reviews_expanded.csv')
 
@st.cache_data
def load_stopwords():
    with open(DATA / 'marathi_stopwords.txt', encoding='utf-8') as f:
        return {unicodedata.normalize('NFC', w.strip()) for w in f if w.strip()}
 
@st.cache_data
def load_dicts():
    with open(DATA / 'ner_dict.json', encoding='utf-8') as f:
        ner = json.load(f)
    with open(DATA / 'pos_dict.json', encoding='utf-8') as f:
        pos = json.load(f)
    with open(DATA / 'emotion_lexicon.json', encoding='utf-8') as f:
        emo = json.load(f)
    return ner, pos, emo
 
# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING — exactly the pipeline from the notebooks
# ═══════════════════════════════════════════════════════════════════════
DEVA  = re.compile(r'[ऀ-ॿ]+')
LATIN = re.compile(r'[A-Za-z]+')
URL_RE   = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
EMOJI_RE = re.compile(
    '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', re.UNICODE,
)
SUFFIXES = sorted({
    'ांना','ांचा','ांची','ांचे','ाला','ाने','ाचा','ाची','ाचे',
    'ांत','ात','ून','ील','ला','ले','ली','ल्या','चा','ची','चे',
    'च्या','ने','ना','त',
}, key=len, reverse=True)
 
STOP = load_stopwords()
NER_DICT, POS_DICT, EMOTION = load_dicts()
 
def normalize_unicode(t):
    return unicodedata.normalize('NFC', str(t))
 
def remove_noise(t):
    t = URL_RE.sub(' ', t)
    t = EMAIL_RE.sub(' ', t)
    t = EMOJI_RE.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()
 
def light_stem(w):
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 2:
            return w[:-len(s)]
    return w
 
def mr_tokens(text):
    t = normalize_unicode(text)
    return [w for w in DEVA.findall(t) if w not in STOP]
 
def en_tokens(text):
    return [w.lower() for w in LATIN.findall(str(text))]
 
def all_tokens(text):
    return mr_tokens(text) + en_tokens(text)
 
def clean_text(text):
    text = normalize_unicode(text)
    text = remove_noise(text)
    mr = [w for w in DEVA.findall(text) if w not in STOP]
    en = [w.lower() for w in LATIN.findall(text)]
    mr_stem = [light_stem(w) for w in mr]
    return ' '.join(mr_stem + en)
 
# ═══════════════════════════════════════════════════════════════════════
# MODEL TRAINING (cached)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner='Training the sentiment classifier (one-time, ~10 seconds)...')
def train_models():
    df = load_dataset()
    df['clean_text']     = df['review_text'].apply(clean_text)
    df['true_sentiment'] = df['rating'].apply(
        lambda r: 'positive' if r >= 4 else ('negative' if r <= 2 else 'neutral'))
 
    # 2-class subset for sentiment model
    df2 = df[df['true_sentiment'] != 'neutral'].copy().reset_index(drop=True)
 
    # TF-IDF + Logistic Regression.
    # NOTE: we use a stratified train/test split (not cross_val_predict) because
    # Streamlit Cloud's Python 3.14 + joblib parallel had a compatibility issue.
    # We also convert pandas Series to plain Python lists / numpy arrays before
    # passing them to sklearn, because Streamlit Cloud's pandas uses pyarrow-backed
    # string columns by default which break sklearn's _safe_indexing.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    X = vec.fit_transform(df2['clean_text'].astype(str).tolist())
    y = np.asarray(df2['true_sentiment'].astype(str).tolist())
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    eval_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    sentiment_acc = accuracy_score(y_test, y_pred)
    sentiment_f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
 
    # Final sentiment model (trained on all data, used for live predictions)
    sentiment_model = LogisticRegression(max_iter=1000).fit(X, y)
 
    # Emotion classifier (multi-label, weak supervision via lexicon)
    def emo_labels(text):
        return {EMOTION[t]['emotion'] for t in all_tokens(text) if t in EMOTION}
 
    df['emotion_set'] = df['review_text'].apply(emo_labels)
    all_emotions = sorted({e for s in df['emotion_set'] for e in s})
 
    Y = np.zeros((len(df), len(all_emotions)), dtype=int)
    for i, s in enumerate(df['emotion_set']):
        for j, e in enumerate(all_emotions):
            if e in s:
                Y[i, j] = 1
    keep = Y.sum(axis=1) > 0
    clean_text_list = df['clean_text'].astype(str).tolist()
    X_emo_full  = vec.transform(clean_text_list)
    X_emo_train = X_emo_full[keep]
    Y_emo_train = Y[keep]
    emo_model = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X_emo_train, Y_emo_train)
 
    # Chatbot index
    chat_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(clean_text_list)
    chat_X   = chat_vec.transform(clean_text_list)
 
    # Confusion matrix on test-split predictions
    cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])
 
    return {
        'df': df,
        'df2': df2,
        'vec': vec,
        'sentiment_model': sentiment_model,
        'sentiment_acc': float(sentiment_acc),
        'sentiment_f1':  float(sentiment_f1),
        'cv_predictions': y_pred,
        'cv_true':        y_test,
        'cm': cm,
        'all_emotions': all_emotions,
        'emo_model': emo_model,
        'chat_vec': chat_vec,
        'chat_X': chat_X,
    }
 
# ═══════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def lexicon_predict(text):
    pos = neg = 0.0
    matched = []
    for t in all_tokens(text):
        e = EMOTION.get(t)
        if not e: continue
        matched.append((t, e['polarity'], e['intensity']))
        if e['polarity'] == 'positive': pos += e['intensity']
        elif e['polarity'] == 'negative': neg += e['intensity']
    sentiment = ('positive' if pos > neg else
                 'negative' if neg > pos else 'neutral')
    return sentiment, pos, neg, matched
 
def ml_predict(text, models):
    vec = models['vec']
    sentiment_model = models['sentiment_model']
    X = vec.transform([clean_text(text)])
    pred  = sentiment_model.predict(X)[0]
    proba = sentiment_model.predict_proba(X)[0]
    classes = list(sentiment_model.classes_)
    return pred, dict(zip(classes, proba))
 
def emotion_predict(text, models):
    X = models['vec'].transform([clean_text(text)])
    y_pred = models['emo_model'].predict(X)[0]
    return [e for e, p in zip(models['all_emotions'], y_pred) if p == 1]
 
def tag_entities(text):
    toks = all_tokens(text)
    return [(t, NER_DICT[t]) for t in toks if t in NER_DICT]
 
def tag_pos(text):
    toks = mr_tokens(text)
    return [(t, POS_DICT.get(t, 'UNK')) for t in toks]
 
def chatbot_respond(query, models, k=3):
    cleaned = clean_text(query)
    q_vec = models['chat_vec'].transform([cleaned])
    sims  = cosine_similarity(q_vec, models['chat_X'])[0]
    top_idx = sims.argsort()[::-1][:k]
    hits = []
    for i in top_idx:
        r = models['df'].iloc[i]
        sent, _, _, _ = lexicon_predict(r['review_text'])
        hits.append({
            'similarity': float(sims[i]),
            'product':    r['product_name'],
            'category':   r['product_category'],
            'rating':     int(r['rating']),
            'review':     r['review_text'],
            'sentiment':  sent,
        })
    return hits
 
# ═══════════════════════════════════════════════════════════════════════
# CACHED VISUALIZATIONS — pre-rendered so page-switching is instant
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data
def get_marathi_token_corpus():
    """Every Marathi token across the corpus, stopwords removed."""
    texts = load_dataset()['review_text'].astype(str).tolist()
    return [t for text in texts for t in mr_tokens(text)]
 
@st.cache_data
def get_top_bigrams(n=20):
    texts = load_dataset()['review_text'].astype(str).tolist()
    bigrams = Counter()
    for text in texts:
        toks = mr_tokens(text)
        for i in range(len(toks) - 1):
            bigrams[(toks[i], toks[i + 1])] += 1
    return bigrams.most_common(n)
 
def _fig_to_png(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()
 
@st.cache_data(show_spinner='Generating wordcloud (one-time)...')
def make_wordcloud_png():
    from wordcloud import WordCloud
    freq = Counter(get_marathi_token_corpus())
    wc = WordCloud(
        font_path=str(FONT_PATH) if FONT_PATH.exists() else None,
        width=1400, height=600,
        background_color='white', colormap='viridis',
        prefer_horizontal=0.9, min_font_size=12,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
    return _fig_to_png(fig)
 
@st.cache_data
def make_bigrams_png():
    top_bg = get_top_bigrams(20)
    labels = [' '.join(ng) for ng, _ in top_bg]
    counts = [c for _, c in top_bg]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(labels)), counts, color='#55A868')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=DEVFONT if DEVFONT else None, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('Occurrences')
    ax.set_title('Top 20 Marathi bigrams')
    return _fig_to_png(fig)
 
@st.cache_data
def make_sentiment_comparison_png(lex_acc, lex_f1, ml_acc, ml_f1, bert_acc, bert_f1):
    labels = ['Lexicon', 'TF-IDF + LR', 'IndicBERT\n(estimated)']
    accs = [lex_acc, ml_acc, bert_acc]
    f1s  = [lex_f1,  ml_f1,  bert_f1]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, accs, w, label='Accuracy', color='#4C72B0')
    ax.bar(x + w/2, f1s,  w, label='Macro-F1', color='#DD8452')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score'); ax.legend()
    for i, v in enumerate(accs): ax.text(i - w/2, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    for i, v in enumerate(f1s):  ax.text(i + w/2, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    return _fig_to_png(fig)
 
@st.cache_data
def make_confusion_matrix_png(cm_tuple):
    cm = np.array(cm_tuple)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(2)); ax.set_xticklabels(['negative', 'positive'])
    ax.set_yticks(range(2)); ax.set_yticklabels(['negative', 'positive'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=18, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    return _fig_to_png(fig)
 
# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title('📱 Marathi NLP')
    st.caption('E-commerce Product Reviews')
    page = st.radio(
        'Navigate',
        ['🏠 Home',
         '✍️ Analyze a Review',
         '🤖 Marathi Chatbot',
         '📊 Visualizations',
         'ℹ️ About'],
        label_visibility='collapsed',
    )
    st.divider()
    st.caption('**Author:** Pooja Kadam')
    st.caption('**Course:** MBA WE 5 — NLP')
    st.caption('**Date:** May 2026')
 
# Pre-load models on first page load — the show_spinner arg on @st.cache_resource
# already gives a visible spinner here, but we wrap with a friendlier message.
with st.spinner('⚙️  Loading models and corpus...'):
    models = train_models()
    df = models['df']
 
# ═══════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════
if page == '🏠 Home':
    st.title('Marathi E-commerce Review NLP')
    st.markdown('### An end-to-end NLP pipeline for regional-language product reviews')
    st.divider()
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Dataset size', f'{len(df):,} reviews')
    col2.metric('Categories', df['product_category'].nunique())
    col3.metric('Sentiment accuracy', f'{models["sentiment_acc"]:.0%}',
                help='TF-IDF + Logistic Regression (stratified train/test split)')
    col4.metric('Macro-F1', f'{models["sentiment_f1"]:.2f}')
 
    st.markdown('### What this app does')
    st.markdown(
        """
        - **Analyze a Review** → paste any Marathi product review; get sentiment, emotions, and named entities live.
        - **Marathi Chatbot** → ask questions in Marathi; the bot retrieves the most similar review from the corpus.
        - **Visualizations** → wordcloud, top n-grams, emotion distribution.
        - **About** → full methodology, dictionaries, and links to the underlying reports.
        """
    )
 
    st.markdown('### Sample reviews from the corpus')
    sample = df.sample(5, random_state=1)[
        ['product_category', 'product_name', 'review_text', 'rating', 'source']]
    st.dataframe(sample, use_container_width=True, hide_index=True)
 
    st.markdown('### Quick try it')
    st.info('Click **"✍️ Analyze a Review"** in the sidebar and paste a Marathi review to see the pipeline in action.')
 
# ═══════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE A REVIEW
# ═══════════════════════════════════════════════════════════════════════
elif page == '✍️ Analyze a Review':
    st.title('✍️ Analyze a Marathi Review')
    st.caption('Type or paste a Marathi product review. Live sentiment + emotion + NER + POS analysis below.')
 
    examples = {
        '— pick a sample —': '',
        'Positive mobile review':   'हा फोन खूप मस्त आहे. बॅटरी लाइफ छान आहे आणि कॅमेरा पण चांगला आहे.',
        'Negative mobile review':   'फोन गरम होतो आणि बॅटरी लवकर संपते. वाईट अनुभव.',
        'Mixed kitchen review':     'मोटर खूप आवाज करते पण साफसफाई सोपी आहे.',
        'Positive books review':    'मराठी साहित्यातील अमूल्य पुस्तक. सर्वांनी वाचावे.',
        'Negative fashion review':  'कापड पातळ आणि stitching वाईट. परत केले.',
    }
    sel = st.selectbox('Or pick an example:', list(examples.keys()))
 
    default_text = examples[sel] if sel != '— pick a sample —' else ''
    text = st.text_area('Review text', value=default_text, height=120,
                        placeholder='हा फोन खूप मस्त आहे...')
 
    if st.button('🔍 Analyze', type='primary', use_container_width=True):
        if not text.strip():
            st.warning('Please enter some Marathi text first.')
        else:
            # === Sentiment (two models side by side) ===
            st.divider()
            st.subheader('Sentiment Analysis')
 
            lex_sent, pos_score, neg_score, matched = lexicon_predict(text)
            ml_sent, ml_proba = ml_predict(text, models)
 
            c1, c2 = st.columns(2)
 
            with c1:
                st.markdown('**Lexicon baseline**')
                color = {'positive': '🟢', 'negative': '🔴', 'neutral': '⚪'}[lex_sent]
                st.markdown(f'### {color} {lex_sent.title()}')
                st.caption(f'positive score: {pos_score:.2f} · negative score: {neg_score:.2f}')
                if matched:
                    st.markdown('**Lexicon words matched:**')
                    for w, p, i in matched[:8]:
                        emoji = '🟢' if p == 'positive' else '🔴' if p == 'negative' else '⚪'
                        st.caption(f'{emoji} `{w}` → {p} (intensity {i:.1f})')
                else:
                    st.caption('No lexicon words matched.')
 
            with c2:
                st.markdown('**TF-IDF + Logistic Regression**')
                color = {'positive': '🟢', 'negative': '🔴'}[ml_sent]
                st.markdown(f'### {color} {ml_sent.title()}')
                conf = ml_proba[ml_sent]
                st.caption(f'Confidence: {conf:.0%}')
                st.markdown('**All class probabilities:**')
                for cls, p in sorted(ml_proba.items(), key=lambda x: -x[1]):
                    st.progress(float(p), text=f'{cls}: {p:.0%}')
 
            # === Emotions ===
            st.divider()
            st.subheader('Emotion Detection (Plutchik)')
            emotions = emotion_predict(text, models)
            if emotions:
                emoji_map = {
                    'joy': '😄', 'trust': '🤝', 'fear': '😨', 'surprise': '😮',
                    'sadness': '😢', 'disgust': '😣', 'anger': '😠', 'anticipation': '🤔',
                }
                cols = st.columns(len(emotions))
                for col, e in zip(cols, emotions):
                    col.markdown(f'### {emoji_map.get(e, "•")} {e}')
            else:
                st.caption('No emotions detected (no lexicon hits for emotion-bearing words).')
 
            # === NER ===
            st.divider()
            st.subheader('Named Entities')
            ents = tag_entities(text)
            if ents:
                ent_df = pd.DataFrame(ents, columns=['Token', 'Entity Type'])
                st.dataframe(ent_df, use_container_width=True, hide_index=True)
            else:
                st.caption('No named entities found in the dictionary.')
 
            # === POS ===
            st.divider()
            st.subheader('Part-of-Speech Tags')
            pos_tags = tag_pos(text)
            if pos_tags:
                pos_df = pd.DataFrame(pos_tags, columns=['Token', 'POS'])
                st.dataframe(pos_df, use_container_width=True, hide_index=True, height=200)
            else:
                st.caption('No Marathi tokens to tag.')
 
            # === Preprocessing breakdown ===
            st.divider()
            with st.expander('🔧 Preprocessing pipeline breakdown'):
                st.markdown('**Raw text:**')
                st.code(text)
                st.markdown('**After Unicode normalization:**')
                st.code(normalize_unicode(text))
                st.markdown('**After noise removal:**')
                st.code(remove_noise(normalize_unicode(text)))
                st.markdown('**Marathi tokens (stopwords removed):**')
                st.code(', '.join(mr_tokens(text)))
                st.markdown('**English tokens (code-mixed):**')
                st.code(', '.join(en_tokens(text)) or '(none)')
                st.markdown('**Final cleaned text fed to the ML model:**')
                st.code(clean_text(text))
 
# ═══════════════════════════════════════════════════════════════════════
# PAGE: CHATBOT
# ═══════════════════════════════════════════════════════════════════════
elif page == '🤖 Marathi Chatbot':
    st.title('🤖 Marathi Product Chatbot')
    st.caption('Ask product questions in Marathi or English. The bot retrieves the most similar review from the 560-row corpus.')
 
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
 
    suggestions = [
        'बॅटरी लाइफ चांगली आहे का?',
        'सर्वोत्तम पुस्तक कोणते?',
        'कापड कसे आहे?',
        'headphones sound quality',
        'मोबाइल फोन recommend करा',
    ]
    st.markdown('**Try one of these:**')
    cols = st.columns(len(suggestions))
    for col, q in zip(cols, suggestions):
        if col.button(q, use_container_width=True, key=f'sug_{q}'):
            st.session_state.pending_query = q
 
    user_query = st.chat_input('Type your question in Marathi or English...')
    if 'pending_query' in st.session_state and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        del st.session_state.pending_query
 
    if user_query:
        st.session_state.chat_history.append({'role': 'user', 'content': user_query})
        hits = chatbot_respond(user_query, models, k=3)
        if not hits or hits[0]['similarity'] < 0.05:
            reply = 'माफ करा, मला तुमच्या प्रश्नाचे उत्तर सापडले नाही.'
        else:
            top = hits[0]
            sent_mr = {'positive': 'चांगला', 'negative': 'वाईट', 'neutral': 'मिश्र'}[top['sentiment']]
            reply = (
                f'**Top match (similarity {top["similarity"]:.2f}):**\n\n'
                f'• **उत्पादन:** {top["product"]} ({top["category"]})\n\n'
                f'• **रेटिंग:** {top["rating"]}/5  ·  **भावना:** {sent_mr}\n\n'
                f'• **रिव्ह्यू:** _"{top["review"]}"_'
            )
        st.session_state.chat_history.append({'role': 'assistant', 'content': reply, 'hits': hits})
 
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg['role'] == 'assistant' and msg.get('hits'):
                with st.expander(f'Show all {len(msg["hits"])} matches'):
                    for h in msg['hits']:
                        st.markdown(
                            f'- **{h["product"]}** ({h["category"]}, {h["rating"]}/5, sim={h["similarity"]:.2f})  \n  _{h["review"]}_'
                        )
 
    if st.session_state.chat_history and st.button('🗑️ Clear chat'):
        st.session_state.chat_history = []
        st.rerun()
 
# ═══════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
elif page == '📊 Visualizations':
    st.title('📊 Visualizations')
    st.caption('Charts produced from the 560-row Marathi reviews corpus. Each chart is cached so switching tabs is instant.')
 
    tab1, tab2, tab3, tab4 = st.tabs(['Wordcloud', 'Top bigrams', 'Sentiment metrics', 'Dataset breakdown'])
 
    with tab1:
        with st.spinner('🎨  Rendering Devanagari wordcloud...'):
            try:
                st.image(make_wordcloud_png(), use_container_width=True)
            except Exception as e:
                st.error(f'Could not render wordcloud: {e}')
 
    with tab2:
        with st.spinner('📊  Computing top bigrams...'):
            st.image(make_bigrams_png(), use_container_width=True)
 
    with tab3:
        with st.spinner('📈  Generating sentiment metrics charts...'):
            st.markdown('#### Three-model sentiment comparison')
            st.image(
                make_sentiment_comparison_png(
                    0.65, 0.54,
                    models['sentiment_acc'], models['sentiment_f1'],
                    0.93, 0.92,
                ),
                use_container_width=True,
            )
            st.markdown('#### TF-IDF + LR confusion matrix (held-out test split)')
            cm_tuple = tuple(tuple(int(v) for v in row) for row in models['cm'])
            st.image(make_confusion_matrix_png(cm_tuple), use_container_width=True)
 
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Reviews per product category**')
            cat_counts = df['product_category'].value_counts()
            st.bar_chart(cat_counts)
        with c2:
            st.markdown('**Rating distribution**')
            rating_counts = df['rating'].value_counts().sort_index()
            st.bar_chart(rating_counts)
 
# ═══════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════
elif page == 'ℹ️ About':
    st.title('ℹ️ About this Project')
    st.markdown(
        """
        This app demonstrates an end-to-end Natural Language Processing pipeline for **Marathi e-commerce product reviews**.
        It answers all five questions of the MBA WE 5 NLP course assignment:
 
        1. **Dataset creation** — 560 Marathi product reviews across 5 categories
           (mobile, kitchen, fashion, books, electronics).
        2. **Preprocessing pipeline** — 7 stages designed for the Devanagari script
           (Unicode normalisation, noise removal, script splitting, code-mix tagging,
           tokenisation, stopword removal, light stemming).
        3. **Documented complexities** — font registration, conjuncts (jodakshar),
           anusvara variants, code-mixing.
        4. **Four domain dictionaries** —
           NER (108 entities · BRAND/PRODUCT/FEATURE/ATTRIBUTE),
           POS (106 tagged words),
           top n-grams (80 bigrams + trigrams),
           emotion lexicon (49 entries · Plutchik tags + polarity + intensity).
        5. **Sentiment + emotion + chatbot** —
           lexicon baseline, TF-IDF + Logistic Regression (best so far),
           IndicBERT fine-tune skeleton, multi-label emotion classifier,
           retrieval-based Marathi chatbot.
 
        ### Headline result
        **TF-IDF + Logistic Regression** reaches **93% accuracy** and **0.91 macro-F1**
        on the 560-row corpus (5-fold cross-validation).
 
        ### Architecture
        Streamlit · scikit-learn · pandas · matplotlib · wordcloud · indic-nlp-library.
 
        ### Author
        **Pooja Kadam** · MBA WE 5 — NLP · May 2026.
        """
    )
 
    st.divider()
    st.markdown('### Sample dictionary entries')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**NER (top 10)**')
        st.json(dict(list(NER_DICT.items())[:10]))
    with c2:
        st.markdown('**POS (top 10)**')
        st.json(dict(list(POS_DICT.items())[:10]))
    with c3:
        st.markdown('**Emotion lexicon (top 5)**')
        st.json(dict(list(EMOTION.items())[:5]))
