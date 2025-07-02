# predict_vn.py
import re
import json
import joblib
import numpy as np

# === CONFIGURATION ===
VEC_PATH = "vectorizer_vn.joblib"
MODEL_PATHS = {
    "Naive Bayes": "nb_model_vn.joblib",
    "Logistic Regression": "lr_model_vn.joblib",
    "SVM": "svm_model_vn.joblib"
}
STOPWORD_PATH = "vn_stopword.json"
MAP_LABEL = {2: "positive", 1: "neutral", 0: "negative"}

# === TEXT CLEANING ===
def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    first_key = next(iter(data))
    return set(data[first_key])

def preprocess(text, stopwords):
    text = text.lower()
    text = re.sub(r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                  r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
                  r"ùúụủũưừứựửữỳýỵỷỹđ\s]", "", text)
    tokens = [tok for tok in text.split() if tok not in stopwords]
    return " ".join(tokens), tokens

# === MODEL HELPERS ===
def get_weights(model):
    if hasattr(model, "feature_log_prob_"):  # Naive Bayes
        return model.feature_log_prob_
    if hasattr(model, "coef_"):  # Logistic Regression, SVM
        return model.coef_
    return None

def top_word_contributions(vectorizer, model, tokens, label_id):
    weights = get_weights(model)
    if weights is None:
        return []

    vocab = vectorizer.vocabulary_
    word_weights = []

    try:
        class_index = list(model.classes_).index(label_id)
    except:
        class_index = 0  # fallback for binary model

    if weights.ndim == 1:
        class_weights = weights
    else:
        class_weights = weights[class_index]

    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            score = class_weights[idx]
            word_weights.append((tok, score))

    return sorted(word_weights, key=lambda x: -abs(x[1]))[:5]

# === PREDICT & EXPLAIN ===
def predict_all_models(raw_input):
    stopwords = load_stopwords(STOPWORD_PATH)
    vectorizer = joblib.load(VEC_PATH)
    cleaned_text, tokens = preprocess(raw_input, stopwords)
    vec = vectorizer.transform([cleaned_text])

    print(f"\n📥 Input: {raw_input}")
    print(f"🧹 Cleaned: {cleaned_text}")
    print(f"🔤 Tokens: {tokens}")

    for name, path in MODEL_PATHS.items():
        model = joblib.load(path)
        label_id = model.predict(vec)[0]
        label_name = MAP_LABEL.get(label_id, str(label_id))

        print(f"\n🔮 {name} Prediction: {label_name}")
        print("🔍 Top contributing words:")

        for word, score in top_word_contributions(vectorizer, model, tokens, label_id):
            print(f"  {word}: {score:.4f}")

# === MAIN ===
if __name__ == "__main__":
    user_input = input("Nhập câu tiếng Việt: ")
    predict_all_models(user_input)
