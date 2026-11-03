"""
app.py - SpamShield Streamlit dashboard.

Run with:   streamlit run app.py

The app only LOADS the model saved by train_model.py; it never retrains.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import clean_text  # the SAME cleaning function used in training

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

# ---- Colour palette (dark navy + blue/cyan accents) ------------------------
NAVY = "#0A1128"
PANEL = "#111C3A"
BORDER = "#1F2F5C"
BLUE = "#3B82F6"
CYAN = "#22D3EE"
TEXT = "#E6EDF7"
MUTED = "#8FA3C7"
SPAM_RED = "#F87171"
HAM_GREEN = "#34D399"

SAMPLE_MESSAGES = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 now to claim. Claim code KL341.",
    "Free offer": "Free entry in 2 a weekly competition to win FA Cup tickets! Text FA to 87121 to receive entry. T&C apply 08452810075",
    "Urgent alert": "URGENT! Your mobile number has won £2000 bonus caller prize. Call 09050000460 from land line. Claim code 3030 valid 12hrs",
    "Cheap deal": "Congratulations! You've been chosen for a free $500 gift card. Visit www.claim-prize-now.com before midnight.",
    "Plans with a friend": "Hey, are we still meeting at the canteen after the lecture? I'll save you a seat.",
    "Study message": "Can you send me the notes from today's data science class? I missed the last half hour.",
    "Reminder": "Don't forget to bring your laptop charger tomorrow, the lab has very few sockets.",
    "Casual chat": "Haha that was so funny. Call me when you reach home, mum is asking about dinner.",
}

st.set_page_config(page_title="SpamShield", page_icon="🛡️", layout="wide")

# ---- Styling ---------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'Manrope', sans-serif; }}
    .stApp {{ background: {NAVY}; color: {TEXT}; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}
    h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; }}
    p, li, label, .stMarkdown {{ color: {TEXT}; }}

    .hero {{ padding: 8px 0 18px 0; }}
    .hero h1 {{ font-size: 3rem; font-weight: 800; margin: 0; }}
    .hero h1 span {{ background: linear-gradient(90deg, {BLUE}, {CYAN});
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .hero p {{ color: {MUTED}; font-size: 1.05rem; max-width: 720px; margin-top: 6px; }}

    .kpi {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
            padding: 18px 20px; border-top: 3px solid {CYAN}; }}
    .kpi .value {{ font-size: 2rem; font-weight: 800; color: {TEXT}; line-height: 1.1; }}
    .kpi .label {{ color: {MUTED}; font-size: 0.85rem; margin-top: 4px; }}

    .result {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
               padding: 22px 24px; margin-top: 8px; }}
    .result .verdict {{ font-size: 2.2rem; font-weight: 800; margin: 0; }}
    .result .sub {{ color: {MUTED}; margin-top: 2px; }}
    .bar-track {{ background: {NAVY}; border-radius: 8px; height: 12px; margin-top: 14px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 8px; }}

    .stTextArea textarea {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 10px; }}
    .stTextArea textarea:focus {{ border-color: {CYAN}; box-shadow: 0 0 0 1px {CYAN}; }}
    .stButton > button {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 8px; }}
    .stButton > button:hover {{ border-color: {CYAN}; color: {CYAN}; }}
    .stButton > button:focus-visible {{ outline: 2px solid {CYAN}; }}
    .stButton > button[kind="primary"] {{ background: linear-gradient(90deg, {BLUE}, {CYAN});
                                          color: {NAVY}; border: none; font-weight: 800; }}
    .stTabs [data-baseweb="tab"] {{ color: {MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {CYAN}; }}
    .note {{ color: {MUTED}; font-size: 0.85rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Loading ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the saved model, vectorizer and metrics once (cached between reruns)."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)
    return model, vectorizer, metrics


def style_axes(ax):
    """Make a Matplotlib axis match the dark theme."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def new_figure(width=5, height=3.4):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(PANEL)
    style_axes(ax)
    return fig, ax


def kpi_card(column, value, label):
    column.markdown(
        f'<div class="kpi"><div class="value">{value}</div><div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# ---- Prediction ------------------------------------------------------------
def analyze(message, model, vectorizer):
    """Return a dict with the prediction, or None if nothing useful to analyze."""
    cleaned = clean_text(message)
    if not cleaned:
        return None
    features = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(features)[0]       # [P(ham), P(spam)]
    label = int(np.argmax(probabilities))

    # Which words pushed the decision? tf-idf weight x (log P(w|spam) - log P(w|ham))
    vocabulary = vectorizer.get_feature_names_out()
    log_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    present = features.nonzero()[1]
    influence = sorted(
        ((vocabulary[i], float(features[0, i] * log_ratio[i])) for i in present),
        key=lambda pair: pair[1],
    )
    return {
        "label": label,
        "confidence": float(probabilities[label]),
        "p_ham": float(probabilities[0]),
        "p_spam": float(probabilities[1]),
        "cleaned": cleaned,
        "ham_words": [(w, s) for w, s in influence if s < 0][:5],
        "spam_words": [(w, s) for w, s in reversed(influence) if s > 0][:5],
        "known_terms": len(present),
    }


def show_result(result):
    is_spam = result["label"] == 1
    colour = SPAM_RED if is_spam else HAM_GREEN
    verdict = "🚨 SPAM" if is_spam else "✅ HAM (Not Spam)"
    st.markdown(
        f"""
        <div class="result" style="border-left: 6px solid {colour};">
            <p class="verdict" style="color:{colour};">{verdict}</p>
            <div class="sub">Model confidence: <b>{result['confidence']:.1%}</b></div>
            <div class="bar-track"><div class="bar-fill"
                 style="width:{result['confidence']*100:.1f}%; background:{colour};"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Confidence is the model's own estimate, not a guaranteed probability of being "
        "correct. Naive Bayes scores tend to be over-confident."
    )

    c1, c2 = st.columns(2)
    c1.metric("P(Spam)", f"{result['p_spam']:.1%}")
    c2.metric("P(Ham)", f"{result['p_ham']:.1%}")

    if result["known_terms"] == 0:
        st.warning("None of the words in this message were seen during training, "
                   "so the prediction mostly reflects the training class ratio.")

    with st.expander("See why the model decided this"):
        st.write("**Message after preprocessing:**")
        st.code(result["cleaned"], language=None)
        left, right = st.columns(2)
        left.write("**Words pushing toward Spam**")
        left.write(", ".join(w for w, _ in result["spam_words"]) or "—")
        right.write("**Words pushing toward Ham**")
        right.write(", ".join(w for w, _ in result["ham_words"]) or "—")


# ---- Charts ----------------------------------------------------------------
def confusion_chart(metrics):
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = new_figure(4.6, 3.6)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
                annot_kws={"size": 16, "weight": "bold"}, linewidths=2, linecolor=PANEL)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    return fig


def metrics_chart(metrics):
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]
    fig, ax = new_figure(4.6, 3.6)
    bars = ax.bar(names, values, color=[BLUE, CYAN, BLUE, CYAN])
    ax.set_ylim(0, 1.1)
    ax.set_title("Performance metrics")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}",
                ha="center", color=TEXT, fontsize=10)
    return fig


def class_chart(metrics):
    fig, ax = new_figure(4.6, 3.0)
    ax.barh(["Spam", "Ham"], [metrics["n_spam"], metrics["n_ham"]], color=[SPAM_RED, HAM_GREEN])
    ax.set_title("Messages per class (after cleaning)")
    ax.set_xlabel("Number of messages")
    return fig


def words_chart(words, title, colour):
    fig, ax = new_figure(4.6, 3.4)
    labels = [item["word"] for item in words][::-1]
    scores = [item["score"] for item in words][::-1]
    ax.barh(labels, scores, color=colour)
    ax.set_title(title)
    ax.set_xlabel("Strength of association")
    return fig


# ---- Page ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Spam<span>Shield</span></h1>
      <p>Paste an SMS message and a Multinomial Naive Bayes model, trained on TF-IDF features,
      will tell you whether it looks like spam or a normal (ham) message.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists() and METRICS_PATH.exists()):
    st.error("The trained model was not found.")
    st.info("Run `python train_model.py` once in your terminal (see README), "
            "then refresh this page.")
    st.stop()

try:
    model, vectorizer, metrics = load_artifacts()
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.info("Delete the files in the `models/` folder and run `python train_model.py` again.")
    st.stop()

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, f"{metrics['accuracy']:.1%}", "Accuracy")
kpi_card(k2, f"{metrics['precision']:.1%}", "Precision (spam)")
kpi_card(k3, f"{metrics['recall']:.1%}", "Recall (spam)")
kpi_card(k4, f"{metrics['f1']:.1%}", "F1-score (spam)")
kpi_card(k5, f"{metrics['n_total']:,}", "Messages in dataset")
st.write("")

tab_analyze, tab_perf, tab_how = st.tabs(["Analyze Message", "Model Performance", "How It Works"])

# ---- Tab 1: analyzer -------------------------------------------------------
with tab_analyze:
    if "message_input" not in st.session_state:
        st.session_state["message_input"] = ""
    if "analyzed_text" not in st.session_state:
        st.session_state["analyzed_text"] = None

    def load_sample(text):
        """Callback for sample buttons: fill the text box."""
        st.session_state["message_input"] = text
        st.session_state["analyzed_text"] = None

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Message analyzer")
        st.text_area("Message", key="message_input", height=170,
                     placeholder="Type or paste an SMS message here...",
                     label_visibility="collapsed")
        if st.button("Analyze Message", type="primary", use_container_width=True):
            st.session_state["analyzed_text"] = st.session_state["message_input"]

        text_to_show = st.session_state["analyzed_text"]
        if text_to_show is not None:
            if not text_to_show.strip():
                st.warning("Please enter a message first.")
            else:
                outcome = analyze(text_to_show, model, vectorizer)
                if outcome is None:
                    st.warning("That message has no usable words after cleaning "
                               "(for example, only symbols). Try a longer message.")
                else:
                    show_result(outcome)

    with right:
        st.subheader("Try a sample")
        st.markdown('<p class="note">Click a sample to load it, then press Analyze.</p>',
                    unsafe_allow_html=True)
        for title, text in SAMPLE_MESSAGES.items():
            st.button(title, key=f"sample_{title}", on_click=load_sample, args=(text,),
                      use_container_width=True)

# ---- Tab 2: performance ----------------------------------------------------
with tab_perf:
    st.subheader("How well does the model do?")
    st.markdown(
        f'<p class="note">All numbers below were calculated on {metrics["n_test"]:,} test messages '
        f'the model never saw during training (training set: {metrics["n_train"]:,} messages). '
        "Spam is treated as the positive class.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.pyplot(confusion_chart(metrics))
    c2.pyplot(metrics_chart(metrics))

    st.markdown("#### What the metrics mean")
    st.markdown(
        """
- **Accuracy** - out of all test messages, the share classified correctly. It can be misleading
  here: if only ~13% of messages are spam, a model that always says "ham" would still score ~87%.
- **Precision** - when the model says *spam*, how often is it right? Low precision means real
  messages get wrongly blocked (false positives).
- **Recall** - out of all real spam, how much did the model catch? Low recall means spam
  slips through (false negatives).
- **F1-score** - a single number balancing precision and recall (their harmonic mean).
- **Confusion matrix** - a 2x2 table of correct and wrong predictions. Top-left = ham correctly
  kept, bottom-right = spam correctly caught, top-right = false alarms, bottom-left = missed spam.
        """
    )

    st.markdown("#### Classification report")
    report_rows = {name: values for name, values in metrics["classification_report"].items()
                   if isinstance(values, dict)}  # skips the scalar "accuracy" entry
    report = pd.DataFrame(report_rows).T.round(3)
    report["support"] = report["support"].astype(int)
    st.dataframe(report, use_container_width=True)
    st.caption(f"Overall accuracy: {metrics['accuracy']:.3f}")

    st.markdown("#### What the model learned")
    d1, d2, d3 = st.columns(3)
    d1.pyplot(class_chart(metrics))
    d2.pyplot(words_chart(metrics["top_spam_words"], "Top spam indicators", SPAM_RED))
    d3.pyplot(words_chart(metrics["top_ham_words"], "Top ham indicators", HAM_GREEN))
    st.caption("Words like 'numtoken' and 'currencytoken' are placeholders our preprocessing "
               "inserts for numbers and currency symbols.")

# ---- Tab 3: how it works ---------------------------------------------------
with tab_how:
    st.subheader("The pipeline in six steps")
    st.markdown(
        """
1. **Dataset** - labelled SMS messages (spam or ham).
2. **Preprocessing** - lowercase, replace URLs/numbers/currency symbols with marker words,
   tokenize into words, strip punctuation, remove stopwords.
3. **TF-IDF** - turn each cleaned message into numbers. A word scores high if it appears often
   in this message but rarely across all messages, so distinctive words like "prize" count more
   than common ones.
4. **Training** - Multinomial Naive Bayes learns how likely each word is in spam versus ham.
5. **Prediction** - your message goes through the *same* preprocessing and TF-IDF vectorizer,
   then the model compares the spam and ham scores.
6. **Evaluation** - the model is tested on held-out messages to compute honest metrics.
        """
    )
    st.subheader("Limitations")
    st.markdown(
        """
- Trained on a small, older SMS dataset (mostly British English, roughly 2000s-era spam).
  Modern phishing, WhatsApp or non-English spam may be missed.
- Naive Bayes assumes words are independent, so it ignores word order and context.
- Confidence scores are not calibrated probabilities of being correct.
- This is a learning project, **not a production-ready spam filter**.
        """
    )
