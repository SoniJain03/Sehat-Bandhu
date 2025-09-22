import os
import pickle
import pandas as pd
from deep_translator import GoogleTranslator
import gdown

# -----------------------------
# Auto-download Models if missing
# -----------------------------
files = {
    "random_forest_model.pkl": "https://drive.google.com/file/d/1lmgcWD-lMI9HYFw3diSSLoMkz68I_QQe/view?usp=sharing",
    "symptom_columns.pkl": "https://drive.google.com/file/d/1kmzqeT6NHmluq5d5t5ceJsOeXc621OZs/view?usp=sharing",
    "nlp_disease_model.pkl": "https://drive.google.com/file/d/1_Wj61jOP9bdGniqKo0sQkWoA9knYcEPJ/view?usp=sharing",
    "nlp_vectorizer.pkl": "https://drive.google.com/file/d/1v4HhITVpj7XTfZ7H-LpiATIzQjzCXKhw/view?usp=sharing",
    "hospital_recommender.pkl": "https://drive.google.com/file/d/1xIUh3jFHxfPBUS6N5TmjAYBFdUaxO5KK/view?usp=sharing",
}

for fname, url in files.items():
    if not os.path.exists(fname):
        print(f"Downloading {fname} from Drive...")
        gdown.download(url, fname, quiet=False)

# -----------------------------
# Load Models
# -----------------------------
with open("random_forest_model.pkl", "rb") as f:
    symptom_model = pickle.load(f)

with open("symptom_columns.pkl", "rb") as f:
    symptom_columns = pickle.load(f)

with open("nlp_disease_model.pkl", "rb") as f:
    nlp_model = pickle.load(f)

with open("nlp_vectorizer.pkl", "rb") as f:
    nlp_vectorizer = pickle.load(f)

with open("hospital_recommender.pkl", "rb") as f:
    hospital_recommender = pickle.load(f)

# -----------------------------
# Malayalam → English Symptom Dictionary
# -----------------------------
mal_to_eng = {
    "പനി": "fever",
    "ചുമ": "cough",
    "തലവേദന": "headache",
    "തൊണ്ടവേദന": "sore throat",
    "തലകറക്കം": "dizziness",
    # Add more as per your dataset
}

# -----------------------------
# Functions
# -----------------------------
def predict_symptom_disease(symptom_dict):
    """Predict disease from structured symptom inputs."""
    input_vector = [symptom_dict.get(col, 0) for col in symptom_columns]
    prediction = symptom_model.predict([input_vector])[0]
    return prediction


def predict_nlp_disease(text, language="en"):
    """Predict disease from free-text input (English or Malayalam)."""
    # Translate Malayalam → English if needed
    if language == "ml":
        text = GoogleTranslator(source="ml", target="en").translate(text)

    # Extract only known symptoms
    extracted_symptoms = []
    for eng_sym in symptom_columns:
        if eng_sym in text.lower():
            extracted_symptoms.append(eng_sym)

    if extracted_symptoms:
        text = " ".join(sorted(extracted_symptoms))

    X_vec = nlp_vectorizer.transform([text])
    prediction = nlp_model.predict(X_vec)[0]
    return prediction


def recommend_hospital(specialty=None, district=None, cost=None, top_n=5):
    """Recommend hospitals based on filters."""
    return hospital_recommender.recommend(
        specialty=specialty, district=district, cost=cost, top_n=top_n
    )

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Symptom model test
    test_symptoms = {"fever": 1, "cough": 1, "headache": 0}
    print("Symptom model prediction:", predict_symptom_disease(test_symptoms))

    # NLP model test
    print("NLP prediction (EN):", predict_nlp_disease("I have fever and cough", language="en"))
    print("NLP prediction (ML):", predict_nlp_disease("എനിക്ക് പനി കൂടാതെ ചുമയുണ്ട്", language="ml"))

    # Hospital recommendation test
    print(recommend_hospital(specialty="Cardiology", district="Ernakulam", cost="Paid/Private"))
