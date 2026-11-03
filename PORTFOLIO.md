# SpamShield — Portfolio & Presentation Kit

## Project description (2–3 lines)

SpamShield is a machine-learning web app that classifies SMS messages as spam or ham using TF-IDF text features and a Multinomial Naive Bayes model. It shows a confidence score, the words that influenced each decision, and evaluation metrics (accuracy, precision, recall, F1, confusion matrix) calculated on a held-out test set, all in a Streamlit dashboard.

## Technology tags

`Python` `Pandas` `NumPy` `Scikit-learn` `NLP` `TF-IDF` `Naive Bayes` `Streamlit` `Matplotlib` `Seaborn`

## Key features

- Message analyzer with Spam/Ham verdict and confidence bar
- "Why this decision?" view showing the cleaned text and the most influential words
- Shared preprocessing function, so training and prediction match exactly
- Model trained once and saved; the app loads it instantly instead of retraining
- Real evaluation: accuracy, precision, recall, F1, confusion matrix, classification report
- Dark-navy dashboard with KPI cards, charts and one-click sample messages
- Graceful handling of empty input, missing model files and missing dataset

---

## Short explanation for your teacher (about 1 minute)

"My project, SpamShield, detects spam SMS messages using machine learning. I used the public SMS Spam Collection dataset of roughly 5,500 labelled messages. First I clean the text: lowercase it, replace numbers, links and currency symbols with marker words, split it into words, remove punctuation and remove stopwords. Then I use TF-IDF to convert each message into numbers, so words that are distinctive, like 'prize' or 'claim', get more weight than common words. I train a Multinomial Naive Bayes classifier on 80% of the data and test it on the remaining 20%, which the model has never seen. I measure accuracy, precision, recall, F1 and a confusion matrix. Precision and recall matter because the dataset is imbalanced: only about 13% of messages are spam. The trained model is saved, and a Streamlit app loads it so a user can paste a message and get a prediction with a confidence score. The same cleaning function is used in training and in the app, so the two always match. This is a learning project: the dataset is small and old, and the confidence score is not a guaranteed probability, so it is not production-ready."

*(Add your real test-set numbers from `train_model.py` when you present.)*

---

## Likely viva questions and answers

**1. Why did you choose Multinomial Naive Bayes?**
It works with word-count or TF-IDF features, trains in a fraction of a second, performs well on small text datasets, and its decisions are easy to explain (each word contributes a spam/ham likelihood). It is a standard baseline for text classification.

**2. What does "naive" mean in Naive Bayes?**
It assumes every word is independent of the others given the class. That's untrue for real language, but the simplification makes the maths tractable and still works surprisingly well in practice.

**3. What is TF-IDF and why not just count words?**
TF is how often a word appears in a message; IDF down-weights words that appear in many messages. TF-IDF therefore highlights distinctive words and reduces the influence of common ones that appear everywhere. Plain counts treat all words equally.

**4. Why do you replace numbers and currency symbols with marker words?**
Exact numbers like "81010" are unique and don't help the model generalise, but spam very often contains phone numbers, prices and links. Replacing them with `numtoken`, `currencytoken` and `urltoken` keeps that useful signal.

**5. Why is accuracy not enough here?**
The dataset is imbalanced (about 13% spam). A model that labels everything "ham" would already be about 87% accurate while catching zero spam. Precision, recall and F1 show how well the spam class is actually handled.

**6. What is the difference between precision and recall? Which matters more for spam filtering?**
Precision = of the messages flagged as spam, how many really are. Recall = of all real spam, how many were caught. Wrongly blocking a genuine message (a precision problem) is usually more costly to a user, but a good filter needs both, which is why we also report F1.

**7. What is a confusion matrix?**
A 2×2 table of actual versus predicted classes: true negatives (ham kept), false positives (ham wrongly flagged), false negatives (spam missed) and true positives (spam caught). All other metrics are calculated from it.

**8. Why did you split the data, and what is data leakage?**
Testing on data the model trained on would give an overly optimistic score. I hold out 20% for testing. Leakage means information from the test set influencing training. I avoid it by fitting the TF-IDF vectorizer on the training set only and merely transforming the test set.

**9. Does the confidence score mean the prediction is that likely to be correct?**
No. It is the model's own probability estimate. Naive Bayes tends to be over-confident because its independence assumption multiplies many word probabilities, so scores drift toward 0% or 100%. It is not a calibrated probability of correctness.

**10. Why does the app not retrain every time?**
Training is done once in `train_model.py` and the model and vectorizer are saved with joblib. The app only loads them (and caches them in Streamlit), which makes startup fast and ensures the model stays the same between runs.

**11. What are the limitations of your project, and how would you improve it?**
The dataset is small, old and mostly English; modern phishing and other languages aren't covered; word order is ignored; misspelled spam like "fr33" can evade it. Improvements: more recent and varied data, cross-validation, comparing with Logistic Regression or SVM, probability calibration, and character n-grams.

**12. How do you make sure training and prediction preprocess text identically?**
Both `train_model.py` and `app.py` import the same `clean_text` function from `utils.py`, and the app uses the saved TF-IDF vectorizer. There's a single source of truth for the cleaning logic.
