"""
utils.py - Shared helpers for SpamShield.

IMPORTANT: Both train_model.py (training) and app.py (prediction) import
`clean_text` from this file. Because they share one function, a message is
always cleaned in exactly the same way during training and prediction.
If the two used different cleaning steps, the model would see "unfamiliar"
text at prediction time and its accuracy would silently drop.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Pre-compiled patterns (faster than re-compiling for every message)
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
CURRENCY_PATTERN = re.compile(r"[£$€]")
NUMBER_PATTERN = re.compile(r"\d+")
WORD_PATTERN = re.compile(r"[a-z]+")


def clean_text(text: str) -> str:
    """Turn a raw SMS into a cleaned string of useful words.

    Steps:
      1. Lowercase          -> "FREE" and "free" become the same word.
      2. Replace URLs, currency symbols and numbers with placeholder words.
         Spam is full of links, prices and phone numbers. The exact value
         does not matter ("81010" vs "80062"), but the *presence* of a
         number or a pound sign does, so we keep a marker word instead.
      3. Tokenization + punctuation removal -> keep only runs of letters a-z.
         This splits the text into word tokens and drops punctuation.
      4. Stopword removal   -> drop very common words ("the", "is", "and")
         that carry little meaning for spam detection.
    """
    text = str(text).lower()
    text = text.replace("'", "")                      # "don't" -> "dont"
    text = URL_PATTERN.sub(" urltoken ", text)
    text = CURRENCY_PATTERN.sub(" currencytoken ", text)
    text = NUMBER_PATTERN.sub(" numtoken ", text)
    tokens = WORD_PATTERN.findall(text)               # tokenization
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)
