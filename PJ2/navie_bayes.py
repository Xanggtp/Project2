import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from laonlp.tokenize import word_tokenize
import stopwordsiso as stopwords


# === CONFIGURATION ===
FILE_PATH = 'Laos.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 6000
STOPWORDS_LO = stopwords.stopwords("lo")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\u0E80-\u0EFF\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS_LO]
    return ' '.join(tokens)

def preprocess_text(text: str) -> str:
    """Lowercase + clean symbols + Lao tokenization."""
    text = text.lower()
    text = re.sub(r'[^\u0E80-\u0EFF\s]', '', text)  # Lao character range
    tokens = word_tokenize(text)
    return ' '.join(tokens)


def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess the CSV data."""
    df = pd.read_csv(path)
    df['text'] = df['text'].astype(str).apply(preprocess_text)
    return df


def vectorize_text(train_texts, test_texts):
    """Convert text to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def train_and_evaluate(X_train, y_train, X_test, y_test, vectorizer, model):
    """Train Naive Bayes and evaluate performance."""
    # Cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold)
    print(f"✅ Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")


    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    # Report
    print("\n📊 Classification Report:")
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


    # Optionally save model
    # joblib.dump(model, "naive_bayes_lao_sentiment.pkl")
    # print("💾 Model saved to naive_bayes_lao_sentiment.pkl")

def count_class_labels(df: pd.DataFrame) -> pd.Series:
        if 'label' not in df.columns:
            raise ValueError("The DataFrame does not contain a 'label' column.")
        return df['label'].value_counts()


def main():
    # Load & preprocess data
    print("📥 Loading data...")
    data = load_data(FILE_PATH)
    X = data['text']
    y = data['label']


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


    # Vectorize text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)


    # Initialize and train model
    model = MultinomialNB(alpha=0.1)
    train_and_evaluate(X_train_vec, y_train, X_test_vec, y_test, vectorizer, model)


    label_counts = count_class_labels(data)
    print("Class label counts:")
    print(label_counts)


if __name__ == "__main__":
    main()
