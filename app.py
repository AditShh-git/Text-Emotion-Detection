import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import requests

MODEL_DIR = "models"

# Dictionary of your models and their Google Drive file IDs (replace with your actual IDs)
MODEL_FILE_IDS = {
    "SVM": "1vlYNNMCaQSTevg4hAuzW7MNCz5VCv1Po",
    "Random Forest": "1E2F4hLMkTRKYrki0mV6-18AJ6L22rRsh",
    "Logistic Regression": "1uKkiWoHFaWTU-Rkm7c4NJ661_i8ZcreW",
    "Naive Bayes": "1zDczUC9fCl_kSWQ5iIsgLwclRLXiqvJG"
}

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

def load_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    models = {}
    for model_name, file_id in MODEL_FILE_IDS.items():
        model_path = os.path.join(MODEL_DIR, f"pipe_{model_name.replace(' ', '_').lower()}.pkl")
        if not os.path.exists(model_path):
            st.info(f"Downloading {model_name} model...")
            try:
                download_file_from_google_drive(file_id, model_path)
            except Exception as e:
                st.error(f"Failed to download {model_name} model: {e}")
                continue
        try:
            models[model_name] = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading {model_name} model: {e}")
    return models

models = load_models()

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

def get_best_model_prediction(text):
    preds = {}
    for model_name, model in models.items():
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0] if hasattr(model, "predict_proba") else None
        confidence = np.max(proba) if proba is not None else 1.0
        preds[model_name] = (pred, proba, confidence)

    best_model = max(preds.items(), key=lambda x: x[1][2])
    model_used, (prediction, proba, confidence) = best_model
    return prediction, proba, confidence, model_used

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions in Text with Multiple Models")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type your text here")
        submit_text = st.form_submit_button(label='Detect Emotion')

    if submit_text and raw_text.strip():
        col1, col2 = st.columns(2)

        prediction, proba_results, confidence, model_used = get_best_model_prediction(raw_text)

        with col1:
            st.success("Input Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction.lower(), "")
            st.write(f"{prediction} {emoji_icon}")
            st.write(f"Confidence: {confidence:.2f}")
            st.info(f"Model Used: {model_used}")

        with col2:
            if proba_results is not None:
                st.success("Prediction Probabilities")
                proba_df = pd.DataFrame([proba_results], columns=models[model_used].classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotion", "Probability"]

                chart = alt.Chart(proba_df_clean).mark_bar().encode(
                    x=alt.X('Emotion:N', sort='-y'),
                    y='Probability:Q',
                    color='Emotion:N'
                ).properties(width=400, height=300)

                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning(f"Model '{model_used}' does not support probability prediction.")

if __name__ == '__main__':
    main()
