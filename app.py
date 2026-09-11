import re
import joblib
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# One-time NLTK setup (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return set(stopwords.words("english")), WordNetLemmatizer()


stop_words, lemmatizer = setup_nltk()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                  # remove HTML
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)                 # remove emails
    text = re.sub(r"[^a-zA-Z\s]", " ", text)             # keep only letters
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Load the trained model + vectorizer (produced by the notebook)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    best_model, tfidf = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

st.title("📧 Spam Email Detector")
st.write("Paste an email below and this NLP model will classify it as **Spam** or **Ham**.")

if not model_loaded:
    st.error(
        "Could not find `spam_classifier.pkl` and `tfidf_vectorizer.pkl` in this folder. "
        "Run the training notebook first, then copy those two files next to `app.py`."
    )
    st.stop()

user_email = st.text_area(
    "Email text",
    height=200,
    placeholder="Paste the email content here..."
)

col1, col2 = st.columns([1, 3])
with col1:
    check_clicked = st.button("Check Email", type="primary")

if check_clicked:
    if not user_email.strip():
        st.warning("Please paste some email text first.")
    else:
        cleaned = preprocess_text(user_email)
        vec = tfidf.transform([cleaned])
        pred = best_model.predict(vec)[0]

        if pred == 1:
            st.error("🚫 **SPAM**")
        else:
            st.success("✅ **HAM (not spam)**")

        # Show confidence if the model supports it
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(vec)[0]
            st.write("**Confidence:**")
            st.progress(float(proba[1]))
            c1, c2 = st.columns(2)
            c1.metric("Ham", f"{proba[0]*100:.1f}%")
            c2.metric("Spam", f"{proba[1]*100:.1f}%")

        with st.expander("See cleaned/preprocessed text"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

st.divider()
st.caption(
    "Model: TF-IDF + classical ML (trained in the companion notebook). "
    "Preprocessing: lowercasing, HTML/URL/email stripping, tokenization, "
    "stopword removal, lemmatization."
)
