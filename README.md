# 🛡️ SpamShield — Intelligent Spam Detection System

A beginner-friendly Machine Learning + NLP project that classifies SMS messages as **Spam** or **Ham** (not spam) and shows a confidence score in a modern Streamlit dashboard.

> **Status:** learning / portfolio project. It is **not production-ready** (see [Limitations](#11-project-limitations)).

---

## 1. Project Overview

SpamShield takes a text message, cleans it, converts it to numbers with TF-IDF, and feeds it to a Multinomial Naive Bayes classifier. The dashboard shows the verdict, the model's confidence, the words that influenced the decision, and evaluation metrics computed on real held-out test data.

## 2. Problem Statement

Spam messages waste time and carry scams and phishing links. Writing manual rules ("block messages containing *free*") is brittle, because spammers change wording and legitimate messages also use those words. A machine-learning model can learn spam patterns from labelled examples instead.

## 3. Objectives

- Build a complete ML pipeline: dataset → preprocessing → TF-IDF → training → prediction → evaluation.
- Use exactly the same preprocessing for training and prediction.
- Report honest metrics calculated from a real test set (nothing hard-coded).
- Present the work in a clean, usable dashboard.

## 4. Technologies Used

| Purpose | Tool |
|---|---|
| Language | Python 3.9+ |
| Data handling | Pandas, NumPy |
| ML & NLP | Scikit-learn (TF-IDF, Multinomial Naive Bayes, metrics) |
| Charts | Matplotlib, Seaborn |
| Web UI | Streamlit |
| Saving models | joblib |

NLTK is **not** used: scikit-learn already ships an English stopword list and Python's `re` module handles tokenization, so adding NLTK would only add a download step.

## 5. Dataset

**SMS Spam Collection** (UCI Machine Learning Repository, also on Kaggle as `spam.csv`): about 5,574 English SMS messages labelled `ham` or `spam`, with roughly 13% spam.

**Getting the data** (the project does *not* ship the file):

- **Automatic:** just run `python train_model.py`. If `data/spam.csv` is missing, the script tries to download it from UCI and saves it for you.
- **Manual (if the download fails, e.g. no internet or a firewall):**
  1. Open <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> (or search "SMS Spam Collection Dataset" on Kaggle).
  2. Download it and save it as `data/spam.csv`.
  3. The loader accepts the Kaggle format (`v1`, `v2` columns), a `label,message` CSV, or the raw UCI tab-separated file.

Cleaning applied: remove empty rows, drop duplicate messages (the raw data has a few hundred), keep only `ham`/`spam` labels.

## 6. How the ML Pipeline Works

### Step 1 — Dataset
A table with a label (`ham`/`spam`) and the message text. Labels become numbers: ham = 0, spam = 1. Spam is the "positive" class for precision and recall.

### Step 2 — Preprocessing (`utils.py → clean_text`)
Computers cannot read text directly, so we first reduce messages to their meaningful words:

| Step | Example | Why |
|---|---|---|
| Lowercase | `FREE` → `free` | "Free" and "FREE" should be the same word |
| Replace URLs / currency / numbers with marker words | `£900` → `currencytoken numtoken` | Exact values don't matter, but *having* a price or phone number is a strong spam clue |
| Tokenization + punctuation removal | `Call now!!!` → `call`, `now` | Split into words and discard symbols |
| Stopword removal | drop `the`, `is`, `and` | Very common words carry little signal |

Because `train_model.py` and `app.py` both import the same `clean_text` function, training and prediction can never drift apart.

### Step 3 — Train/test split
80% of messages train the model; 20% are locked away to test it. `stratify=` keeps the same spam percentage in both parts. `random_state=42` makes the result reproducible.

### Step 4 — TF-IDF
**TF** (term frequency) = how often a word appears in a message. **IDF** (inverse document frequency) = how rare the word is across all messages. Multiplying them gives high scores to words that are frequent in *this* message but rare overall (like "prize") and low scores to common words. Settings used: single words and word pairs (`ngram_range=(1, 2)`), ignore terms seen in fewer than 2 messages (`min_df=2`), and `sublinear_tf=True` to dampen repeated words. The vectorizer is **fitted on training data only**, then used to transform the test data, which avoids data leakage.

### Step 5 — Multinomial Naive Bayes
For each word, the model learns how likely it is in spam versus ham. For a new message it combines those word likelihoods with the overall spam/ham ratio (Bayes' theorem) and picks the more likely class. It's "naive" because it assumes words are independent of each other. It is fast, works well on small text datasets, and is easy to explain. `alpha=0.1` is smoothing so unseen words don't zero out the probability.

### Step 6 — Prediction
The app cleans your message with `clean_text`, transforms it with the saved TF-IDF vectorizer, and calls `predict_proba`. The **confidence** shown is the probability of the predicted class.

### Step 7 — Evaluation
Metrics are computed on the untouched test set and saved to `models/metrics.json`, which the dashboard reads. Nothing is hard-coded.

## 7. Project Structure

```text
SpamShield/
├── app.py              # Streamlit dashboard
├── train_model.py      # Trains, evaluates and saves the model
├── utils.py            # clean_text() shared by training and the app
├── requirements.txt
├── README.md
├── PORTFOLIO.md        # Description, presentation notes, viva Q&A
├── data/
│   └── spam.csv        # you provide this (or train_model.py downloads it)
├── models/             # created by train_model.py
│   ├── spam_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metrics.json    # real test-set metrics shown in the dashboard
└── assets/             # optional: screenshots for your portfolio
```

(`utils.py` is the one addition to the suggested layout. It keeps preprocessing in a single place.)

## 8. Installation (Windows)

1. Install Python 3.9 or newer from <https://www.python.org/downloads/> and tick **"Add Python to PATH"** during setup.
2. Open **Command Prompt** or PowerShell in the project folder (in File Explorer, click the address bar, type `cmd`, press Enter).
3. Create and activate a virtual environment:
   ```bat
   python -m venv venv
   venv\Scripts\activate
   ```
   (In PowerShell, if activation is blocked, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first.)
4. Install the libraries:
   ```bat
   pip install -r requirements.txt
   ```

## 9. How to Run

```bat
:: 1) Train once (downloads the dataset if data\spam.csv is missing)
python train_model.py

:: 2) Start the dashboard
streamlit run app.py
```

Streamlit opens <http://localhost:8501> in your browser. The app only loads the saved model, so it starts instantly and never retrains. Re-run `python train_model.py` only if you change the data or preprocessing.

**Troubleshooting**
- *"The trained model was not found"* → run step 1 first.
- *`streamlit` is not recognized* → activate the virtual environment, or use `python -m streamlit run app.py`.
- *Download failed* → get the dataset manually (see section 5).

## 10. Model Evaluation Explained

After training, the dashboard shows these values, all calculated from the test set (spam = positive class):

| Metric | Plain-language meaning |
|---|---|
| **Accuracy** | Share of all messages classified correctly. Can mislead on imbalanced data (always guessing "ham" already scores ~87%). |
| **Precision** | When the model says "spam", how often is it right? Low precision = real messages wrongly blocked. |
| **Recall** | Of all real spam, how much did we catch? Low recall = spam gets through. |
| **F1-score** | One number balancing precision and recall. |
| **Confusion matrix** | 2×2 table: true negatives, false positives, false negatives, true positives. |
| **Classification report** | Precision, recall and F1 for each class separately, plus averages. |

For a spam filter, **precision matters a lot** (blocking a genuine message is costly), but recall matters too. That's why we look at all four metrics, not just accuracy.

> Record your own numbers after training. Your results come from your run, so quote those in your presentation.

## 11. Project Limitations

- **Small, old, narrow dataset:** ~5.5k messages, mostly British/Singaporean English, collected in the 2000s. Modern phishing, WhatsApp/UPI scams, Hinglish or other languages are not represented and may be misclassified.
- **Naive independence assumption:** word order and context are ignored ("not free" vs "free").
- **Confidence ≠ correctness:** Naive Bayes scores are often over-confident and not calibrated. A 99% score does not mean a 99% chance of being right.
- **Evaluation caveat:** one random 80/20 split on one dataset gives an estimate, not a guarantee. Duplicates and near-duplicates in the data can inflate scores.
- **Easy to evade:** deliberate misspellings (`fr33 pr1ze`) can fool a bag-of-words model.
- **Not production-ready.** Real systems need far more data, monitoring, retraining and human review.

**Possible improvements:** cross-validation, hyper-parameter tuning, comparing with Logistic Regression, probability calibration, character n-grams for misspellings.
