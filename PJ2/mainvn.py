# sentiment_pipeline_vietnamese.py
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# === CONFIGURATION ===========================================================
FILE_PATH     = "Vietnamese.csv"
FILE_STOPWORDS = "vn_stopword.json"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
MAX_FEATURES  = 4000
NGRAM_RANGE   = (1, 2)
MIN_DF        = 2

MAP_LABEL        = {"positive": 2, "negative": 0, "neutral": 1}
REVERT_MAP_LABEL = {v: k for k, v in MAP_LABEL.items()}

# === PREPROCESSING FUNCTIONS =================================================
def load_stopwords(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    first_key = next(iter(data))
    return set(data[first_key])

def preprocess_text(text: str, stopwords: set[str]) -> str:
    text = text.lower()
    # Giữ lại tiếng Việt + dấu + khoảng trắng
    text = re.sub(r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                  r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
                  r"ùúụủũưừứựửữỳýỵỷỹđ\s]", "", text)
    tokens = [tok for tok in text.split() if tok not in stopwords]
    return " ".join(tokens)

def load_data(path: str, stopwords: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).apply(lambda t: preprocess_text(t, stopwords))
    return df

# === VECTORIZATION ===========================================================
def vectorize_text(train_texts, test_texts):
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        analyzer="word",
        token_pattern=r"\S+",
        smooth_idf=True,
    )
    return vec.fit_transform(train_texts), vec.transform(test_texts), vec

# === EVALUATION HELPERS ======================================================
def _get_weights(model):
    if hasattr(model, "feature_log_prob_"):
        return model.feature_log_prob_
    if hasattr(model, "coef_"):
        return model.coef_
    return None

def show_top_words(vectorizer, model, class_labels, top_n=10):
    weights = _get_weights(model)
    if weights is None:
        print("⛔️ Không thể trích xuất trọng số từ mô hình.")
        return
    names = vectorizer.get_feature_names_out()
    for i, lbl in enumerate(class_labels):
        top_idx = np.argsort(weights[i])[::-1][:top_n]
        print(f"\nTop {top_n} từ – {REVERT_MAP_LABEL[lbl]}:")
        for idx in top_idx:
            print(f"  {names[idx]}: {weights[i][idx]:.4f}")

def evaluate(model_name, model, vectorizer, X_train, y_train, X_test, y_test, top_n=15):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv).mean()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} ===")
    print(f"CV Accuracy : {cv_acc:.4f}")
    print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro-F1      : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
            target_names=[REVERT_MAP_LABEL[c] for c in sorted(set(y_test))],
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[REVERT_MAP_LABEL[c] for c in sorted(set(y_test))],
        yticklabels=[REVERT_MAP_LABEL[c] for c in sorted(set(y_test))],
    )
    plt.title(f"{model_name} – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    show_top_words(vectorizer, model, sorted(set(y_test)), top_n)

# === MAIN ====================================================================
def main():
    stopwords = load_stopwords(FILE_STOPWORDS)

    print("📥 Đang tải và tiền xử lý dữ liệu…")
    df = load_data(FILE_PATH, stopwords)
    df = df[df["label"].isin(MAP_LABEL)]  # đảm bảo nhãn hợp lệ
    df["label"] = df["label"].map(MAP_LABEL)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    models = [
        ("MultinomialNB", MultinomialNB(alpha=0.1)),
        (
            "LogisticRegression",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                multi_class="auto",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        ("LinearSVC", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
    ]

    for name, clf in models:
        evaluate(name, clf, vectorizer, X_train_vec, y_train, X_test_vec, y_test)

if __name__ == "__main__":
    main()
