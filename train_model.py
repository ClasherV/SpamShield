"""
train_model.py - Trains the SpamShield model and saves everything app.py needs.

Run once:   python train_model.py
Optional:   python train_model.py --data path/to/other.csv

Pipeline:
    Dataset -> Cleaning -> Train/Test split -> TF-IDF -> Naive Bayes
            -> Evaluation on the unseen test set -> Save to /models
"""

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from utils import clean_text

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

# Official UCI SMS Spam Collection (zip contains a tab-separated file)
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

RANDOM_STATE = 42   # makes the split reproducible
TEST_SIZE = 0.20    # 80% train / 20% test


# --------------------------------------------------------------------------
# 1. DATASET
# --------------------------------------------------------------------------
def download_dataset(destination: Path) -> bool:
    """Try to download the UCI dataset and save it as a CSV. Returns success."""
    try:
        print("Trying to download the SMS Spam Collection from UCI ...")
        with urllib.request.urlopen(UCI_URL, timeout=30) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        raw = archive.read("SMSSpamCollection").decode("utf-8")
        rows = [line.split("\t", 1) for line in raw.splitlines() if "\t" in line]
        df = pd.DataFrame(rows, columns=["label", "message"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)
        print(f"Saved dataset to {destination}")
        return True
    except Exception as error:  # network down, URL changed, etc.
        print(f"Automatic download failed: {error}")
        return False


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV and return a DataFrame with columns: label, message.

    Handles the common formats of this dataset:
      * Kaggle's spam.csv  -> columns v1, v2 (+ empty extra columns), latin-1
      * Our own download   -> columns label, message
      * Raw UCI file       -> tab-separated, no header
    """
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])

    lowered = {col: str(col).strip().lower() for col in df.columns}
    df = df.rename(columns=lowered)

    renames = {"v1": "label", "v2": "message", "category": "label", "text": "message"}
    df = df.rename(columns=renames)

    if not {"label", "message"}.issubset(df.columns):
        # Maybe the file is tab-separated without a header
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"],
                         encoding="latin-1")

    return df[["label", "message"]]


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw table: drop blanks/duplicates, map labels to 0/1."""
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]

    before = len(df)
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate messages.")

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})  # spam = positive class
    df["clean_message"] = df["message"].apply(clean_text)
    # Messages that become empty after cleaning (e.g. only "???") are useless
    df = df[df["clean_message"].str.len() > 0].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 2. TRAINING + EVALUATION
# --------------------------------------------------------------------------
def top_words(model, vectorizer, n: int = 10):
    """Words the model links most strongly to spam and to ham.

    Naive Bayes stores log P(word | class). The difference between the two
    classes shows which words push a message toward spam (positive) or
    toward ham (negative).
    """
    words = np.array(vectorizer.get_feature_names_out())
    difference = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    order = np.argsort(difference)
    spam_idx, ham_idx = order[::-1][:n], order[:n]
    return (
        [{"word": words[i], "score": float(difference[i])} for i in spam_idx],
        [{"word": words[i], "score": float(-difference[i])} for i in ham_idx],
    )


def main(data_path: Path) -> None:
    # --- Load data -------------------------------------------------------
    if not data_path.exists():
        if data_path != DATA_PATH or not download_dataset(data_path):
            print(
                f"\nDataset not found at: {data_path}\n"
                "Download the 'SMS Spam Collection' (see README, section 'Dataset')\n"
                "and save it as data/spam.csv, then run this script again."
            )
            sys.exit(1)

    df = prepare_dataset(load_dataset(data_path))
    print(f"Dataset ready: {len(df)} messages "
          f"({(df.label_num == 0).sum()} ham, {(df.label_num == 1).sum()} spam)")

    # --- Split -----------------------------------------------------------
    # stratify keeps the same spam/ham ratio in both splits (important because
    # spam is only ~13% of the data).
    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_message"], df["label_num"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_num"],
    )

    # --- TF-IDF ----------------------------------------------------------
    # The vectorizer LEARNS its vocabulary from the training set only (fit),
    # then simply converts the test set (transform). Fitting on the test set
    # would be "data leakage".
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} terms")

    # --- Train -----------------------------------------------------------
    model = MultinomialNB(alpha=0.1)  # alpha = smoothing for unseen words
    model.fit(x_train_tfidf, y_train)

    # --- Evaluate on the unseen test set ---------------------------------
    y_pred = model.predict(x_test_tfidf)
    spam_words, ham_words = top_words(model, vectorizer)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True),
        "classification_report_text": classification_report(
            y_test, y_pred, target_names=["Ham", "Spam"]),
        "n_total": int(len(df)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_ham": int((df.label_num == 0).sum()),
        "n_spam": int((df.label_num == 1).sum()),
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "top_spam_words": spam_words,
        "top_ham_words": ham_words,
    }

    # --- Save ------------------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n=== Test-set results (computed, not hard-coded) ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\n" + metrics["classification_report_text"])
    print(f"Saved model to {MODEL_PATH.parent}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SpamShield model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH,
                        help="Path to the SMS dataset CSV (default: data/spam.csv)")
    args = parser.parse_args()
    main(args.data)
