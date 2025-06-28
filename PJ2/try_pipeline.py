
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')
X = df['text'].astype(str)
MAP_LABEL = {'positive': 2, 'negative': 0, 'neutral': 1}
y = df['label'].map(MAP_LABEL)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines
pipeline_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=4000, ngram_range=(1, 1), min_df=2)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=4000, ngram_range=(1, 1), min_df=2)),
    ('clf', SVC(kernel='linear'))
])

# Train and evaluate Logistic Regression
pipeline_logreg.fit(X_train, y_train)
y_pred_logreg = pipeline_logreg.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Train and evaluate SVM
pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
