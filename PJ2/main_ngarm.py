import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from laonlp.tokenize import word_tokenize
import stopwordsiso as stopwords

# === CONFIGURATION ===
FILE_PATH     = 'Laos.csv'
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
MAX_FEATURES  = 4000        # maximum number of TF‑IDF features
NGRAM_RANGE   = (1, 2)      # unigrams + bigrams
MIN_DF        = 2           # ignore terms in fewer than 2 documents

# Load the Lao stop‑word set
STOPWORDS_LO  = stopwords.stopwords("lo")


def preprocess_text(text: str) -> str:
    text = text.lower()
    # keep only Lao unicode range and whitespace
    text = re.sub(r'[^\u0E80-\u0EFF\s]', '', text)
    tokens = word_tokenize(text)
    # filter out stop‑words
    tokens = [t for t in tokens if t not in STOPWORDS_LO]
    return ' '.join(tokens)


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and preprocess the 'text' column."""
    df = pd.read_csv(path)
    df['text'] = df['text'].astype(str).apply(preprocess_text)
    return df


def vectorize_text(train_texts, test_texts):
    """Convert text to TF‑IDF vectors with n‑grams."""
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        analyzer='word',
        token_pattern=r'\S+',
        smooth_idf=True
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def show_top_words(vectorizer, model, class_labels, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    for i, label in enumerate(class_labels):
        top_idx = model.feature_log_prob_[i].argsort()[::-1][:top_n]
        print(f"\nTop {top_n} words for class '{label}':")
        for idx in top_idx:
            print(f"  {feature_names[idx]}: {model.feature_log_prob_[i][idx]:.4f}")


def train_and_evaluate(X_train, y_train, X_test, y_test, vectorizer, model):
    """Cross‑validate, train, evaluate, and plot confusion matrix."""
    # Cross‑validation
    kfold  = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=kfold)
    print(f"Cross‑Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

    # Fit & predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Top‑N words per class
    show_top_words(vectorizer, model, model.classes_)
    TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=2)


def main():
    print("Loading and preprocessing data...")
    data = load_data(FILE_PATH)
    X, y = data['text'], data['label']

    # Train‑test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Vectorize
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    # Initialize model with smoothing parameter
    model = MultinomialNB(alpha=0.1)
    train_and_evaluate(X_train_vec, y_train, X_test_vec, y_test, vectorizer, model)


if __name__ == '__main__':
    main()
