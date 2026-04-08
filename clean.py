

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
PROCESSED_FILENAME = 'final_processed_data.csv'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

lemmatizer = WordNetLemmatizer()
rf_model, svm_model = None, None
metrics_result = {}


# === NLP Utilities ===
def preprocess_text(text):
    text = str(text).lower()                             # Lowercase
    text = re.sub(r'\[.*?\]', '', text)                 # Remove brackets
    text = re.sub(r'[^a-z\s]', '', text)                # Remove special characters
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    text = ' '.join(tokens)
    text = anonymize_text(text)
    return text

def anonymize_text(text):
    text = re.sub(r'\b(name|mr|mrs|ms|dr)\s+\w+\b', 'person', text)
    text = re.sub(r'\d{10}|\d{3}-\d{3}-\d{4}', 'phonenumber', text)
    text = re.sub(r'\b\d{1,3}\b', 'number', text)
    return text

def extract_symptoms(text):
    text = preprocess_text(text)
    return {
        'Fever': int('fever' in text),
        'Cough': int('cough' in text),
        'Fatigue': int('fatigue' in text),
        'Difficulty Breathing': int('difficulty breathing' in text or 'trouble breathing' in text)
    }

def cleanse_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Gender and Age cleanup
    df['Gender'] = df['Gender'].astype(str).str.lower().map({'male': 0, 'female': 1}).fillna(0).astype(int)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].mean()).astype(int)

    # NLP-based symptom extraction from Symptom_Description
    if 'Symptom_Description' in df.columns:
        nlp_features = df['Symptom_Description'].apply(extract_symptoms)
        nlp_df = pd.DataFrame(nlp_features.tolist())
        df = pd.concat([df, nlp_df], axis=1)
    else:
        # Fallback if already structured symptoms
        for col in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']:
            df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # Outcome variable encoding
    df['Outcome Variable'] = df['Outcome Variable'].astype(str).str.lower().map({'positive': 1, 'negative': 0})
    df['Outcome Variable'] = df['Outcome Variable'].fillna(0).astype(int)

    # Save cleaned dataset
    processed_path = os.path.join(PROCESSED_FOLDER, PROCESSED_FILENAME)
    df.to_csv(processed_path, index=False)

    return df


def ensemble_predictions(rf_pred, svm_pred, rf_weight=1, svm_weight=1):
    combined_preds = []
    for r, s in zip(rf_pred, svm_pred):
        votes = Counter()
        votes[r] += rf_weight
        votes[s] += svm_weight
        combined_preds.append(votes.most_common(1)[0][0])
    return combined_preds


def train_models(df):
    global rf_model, svm_model, metrics_result
    features = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Age']
    target = 'Outcome Variable'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Train SVM
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    # Ensemble combined predictions (majority voting)
    combined_pred = ensemble_predictions(rf_pred, svm_pred)

    def compute_metrics(true, pred):
        return {
            "accuracy": round(accuracy_score(true, pred) * 100, 2),
            "precision": round(precision_score(true, pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(true, pred, zero_division=0) * 100, 2),
            "f1Score": round(f1_score(true, pred, zero_division=0) * 100, 2)
        }

    metrics_result = {
        "random_forest": compute_metrics(y_test, rf_pred),
        "svm": compute_metrics(y_test, svm_pred),
        "ensemble": compute_metrics(y_test, combined_pred)  # ensemble metrics added
    }


@app.route('/')
def home():
    return jsonify({"message": "Flask healthcare ML backend with NLP & ensemble models running."})


@app.route('/processed/<filename>')
def serve_processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


@app.route('/metrics')
def get_metrics():
    global metrics_result
    if not metrics_result:
        return jsonify({"error": "Model not trained yet."}), 400
    return jsonify(metrics_result)


if __name__ == '__main__':
    uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    if uploaded_files:
        uploaded_file_path = os.path.join(UPLOAD_FOLDER, uploaded_files[0])
        df = cleanse_and_process_data(uploaded_file_path)
        train_models(df)
    else:
        print("❌ No CSV found in 'uploads' folder.")

    app.run(debug=True)
