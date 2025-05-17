import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

MODEL_DIR = "models"

def load_models(model_dir=MODEL_DIR):
    loaded_models = {}
    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            model_name = file.replace("pipe_", "").replace(".pkl", "").replace("_", " ").title()
            loaded_models[model_name] = joblib.load(os.path.join(model_dir, file))
    return loaded_models

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
