import streamlit as st
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import matplotlib.pyplot as plt

# -------------------------
# Label mapping for baseline model
# -------------------------
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# -------------------------
# Load Baseline Model
# -------------------------
@st.cache_resource
def load_baseline():
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    logreg = joblib.load("baseline_lr_model.joblib")
    return tfidf, logreg

tfidf, logreg = load_baseline()

# -------------------------
# Load DistilBERT Locally
# -------------------------
@st.cache_resource
def load_distilbert():
    tokenizer = AutoTokenizer.from_pretrained("distilbert_sst2")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert_sst2")
    pipeline_model = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline_model

sentiment_model = load_distilbert()

# -------------------------
# Helper Functions
# -------------------------
def predict_baseline(texts):
    return [label_map.get(p, str(p)) for p in logreg.predict(tfidf.transform(texts))]

def predict_distilbert(texts):
    # Accept single string, Series, or list
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    elif isinstance(texts, str):
        texts = [texts]
    return [x['label'] for x in sentiment_model(texts)]

# -------------------------
# Streamlit App UI
# -------------------------
st.title("Patient Feedback Sentiment Analyzer")
st.write("Compare Baseline Model (TF-IDF + Logistic Regression) vs DistilBERT")

# --- Sample Reviews ---
st.subheader("Try Sample Reviews")
sample_reviews = [
    "The doctor was rude.",
    "I had a great experience at the hospital.",
    "Waiting time was too long.",
    "The staff was very helpful and friendly."
]
selected_review = st.selectbox("Choose a sample review", sample_reviews)

if st.button("Use Sample Review"):
    st.session_state['user_text'] = selected_review

user_text = st.text_area("Enter review text:", value=st.session_state.get('user_text', ''), height=80)

if st.button("Predict Sentiment for Text"):
    if user_text.strip():
        baseline_pred = predict_baseline([user_text])[0]
        distilbert_pred = predict_distilbert(user_text)[0]
        st.success(f"Baseline Prediction: {baseline_pred}")
        st.success(f"DistilBERT Prediction: {distilbert_pred}")
    else:
        st.warning("Please enter some text!")

# --- CSV Upload ---
st.header("Upload CSV File for Batch Predictions")
uploaded_file = st.file_uploader("Upload CSV (must have 'text' column)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must have a 'text' column")
    else:
        st.write("Running predictions...")
        df['Baseline'] = predict_baseline(df['text'])
        df['DistilBERT'] = predict_distilbert(df['text'])
        st.write(df.head(10))

        # --- Visualization ---
        st.subheader("Sentiment Distribution Comparison")
        comparison = pd.DataFrame({
            'Baseline': df['Baseline'].value_counts(),
            'DistilBERT': df['DistilBERT'].value_counts()
        }).fillna(0)
        st.bar_chart(comparison)

        # --- Download Predictions ---
        csv = df.to_csv(index=False).encode()
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="sentiment_predictions.csv",
            mime="text/csv"
        )

# --- Download Sample CSV ---
st.header("Download Sample CSV")
sample_df = pd.DataFrame({"text": sample_reviews})
csv_sample = sample_df.to_csv(index=False).encode()
st.download_button(
    label="Download Sample Reviews CSV",
    data=csv_sample,
    file_name="sample_reviews.csv",
    mime="text/csv"
)
