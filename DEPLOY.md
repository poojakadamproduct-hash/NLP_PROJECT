# Deploying the Marathi NLP App

The reference app you shared (`marathi-banking-complaint-sentiment-analyzer-...streamlit.app`) is hosted on **Streamlit Community Cloud** — free hosting designed specifically for Streamlit apps. This guide walks you through deploying **your** app the same way. Total time: **~5 minutes** once you have a GitHub account.

---

## Why Streamlit Cloud (and not Vercel)

| Platform | Supports Streamlit? | Recommendation |
|----------|---------------------|----------------|
| **Streamlit Community Cloud** | ✅ Native, free | ⭐ Use this — same platform as the reference app |
| **Hugging Face Spaces** | ✅ Native, free | Good alternative |
| **Render / Railway** | ✅ Supported, free tier | Good for paid scale-up |
| **Vercel** | ❌ Not natively. Vercel is serverless; Streamlit needs a persistent server | Avoid for Streamlit apps |

If you must use Vercel for some reason, you'd have to rewrite the entire frontend in Next.js and run model inference via serverless Python functions — about 5× the work for the same end-user experience. Stay on Streamlit Cloud unless you have a specific reason not to.

---

## Step 1 — Push the project to GitHub

### Option A: I have Git installed

```bash
cd marathi_nlp_streamlit
git init -b main
git add .
git commit -m "Marathi NLP Streamlit app"
```

Then create a **new public repo** on GitHub (https://github.com/new), name it `marathi-nlp-reviews`, and push:

```bash
git remote add origin https://github.com/<your-username>/marathi-nlp-reviews.git
git push -u origin main
```

### Option B: I want to use the GitHub web UI (no command line)

1. Go to https://github.com/new and create a **public** repo named `marathi-nlp-reviews`.
2. On the new empty repo page, click **uploading an existing file**.
3. Drag and drop **every file and folder** from the unzipped `marathi_nlp_streamlit/` folder into the upload area (do NOT upload the folder itself — upload its contents).
4. Important: the `.streamlit/`, `data/`, and `fonts/` folders must be at the repo root.
5. Click **Commit changes**.

Verify your repo looks like this on GitHub:
```
marathi-nlp-reviews/
├── app.py
├── requirements.txt
├── README.md
├── DEPLOY.md
├── .gitignore
├── .streamlit/config.toml
├── data/  (6 files inside)
└── fonts/ (1 file inside)
```

---

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io/
2. Click **Sign in with GitHub** (free, uses your existing GitHub account).
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in the form:
   - **Repository:** `<your-username>/marathi-nlp-reviews`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL** (optional): pick a memorable subdomain like `marathi-nlp-reviews`
5. Click **Deploy**.
6. Wait 2–4 minutes. Streamlit installs dependencies from `requirements.txt` and launches the app.
7. When the spinner stops, your live URL appears: `https://marathi-nlp-reviews-<random>.streamlit.app/`

That's the same domain pattern as the reference banking-complaint app. **You're live.**

---

## Step 3 — Test the live app

Open your live URL and try each page in the sidebar:

- **🏠 Home** — should show metrics: 560 reviews, 5 categories, ~93% accuracy, 0.91 macro-F1.
- **✍️ Analyze a Review** — pick "Positive mobile review" from the dropdown → click *Analyze*. You should see green sentiment indicators, joy emotion, and 4 named entities.
- **🤖 Marathi Chatbot** — click any suggestion button. The bot replies with a matched review and similarity score.
- **📊 Visualizations** — switch through the 4 tabs. The wordcloud is the showpiece.
- **ℹ️ About** — methodology + sample dictionary entries.

If any page errors, see **Troubleshooting** below.

---

## Step 4 — Share the link

Copy your `https://<your-app>.streamlit.app/` URL and:

- Paste it on your assignment cover sheet.
- Share with faculty over WhatsApp or email.
- Demo live in class (no install needed for the audience).

The link stays alive as long as your GitHub repo exists and Streamlit Cloud is running (free apps occasionally sleep after 7 days of inactivity — visit the URL once a week to keep it warm).

---

## Updating the app after deploy

Edit any file locally → `git commit` → `git push`. Streamlit Cloud detects the push and auto-redeploys within ~1 minute. No manual steps.

---

## Troubleshooting

### "Module not found" on deploy

Open the Streamlit logs (bottom-left "Manage app" → "Logs"). Look for the failing `import` line. Add the missing package to `requirements.txt`, commit, push. Re-deploy happens automatically.

### Devanagari shows as boxes in charts

Confirm `fonts/NotoSansDevanagari-Regular.ttf` is in your GitHub repo and was uploaded as part of the `fonts/` folder. `git lfs` is NOT needed — the file is small enough.

### "Data not found" error on first load

Check that `data/marathi_reviews_expanded.csv` and the other data files are all uploaded to the `data/` folder in the GitHub repo. Streamlit reads them by relative path from `app.py`.

### "App is sleeping" message

Free Streamlit Cloud apps sleep after 7 days of zero traffic. Click **Wake up** on the splash screen and the app is back online within ~30 seconds. Or visit your URL once a week to keep it warm.

### CPU/Memory limits

The free tier gives ~1 GB RAM and 1 CPU. The app's TF-IDF + Logistic Regression model is well under this. If you swap in a heavier transformer model (IndicBERT etc.), you may exceed the free tier and need a paid plan or Hugging Face Spaces instead.

---

## Bonus: deploy to Hugging Face Spaces (alternative)

If you prefer Hugging Face Spaces:

1. Go to https://huggingface.co/new-space
2. Choose **Streamlit** as the SDK.
3. Upload the same files (drag-and-drop in the web UI).
4. Wait ~3 minutes for the build to complete.
5. Your URL: `https://huggingface.co/spaces/<your-username>/marathi-nlp-reviews`

Same UX as Streamlit Cloud, slightly different file-upload flow.

---

🎓 **You're done!** Your live Marathi NLP demo is now publicly accessible. Add the link to your assignment submission and impress your faculty.
