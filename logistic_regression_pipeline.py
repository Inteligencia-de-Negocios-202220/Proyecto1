import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from joblib import dump

# Descargando las stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Función para tokenizar los comentarios
def tokenizer(text):
    tt = TweetTokenizer()
    tokens = tt.tokenize(text)
    return tokens

comments_df = pd.read_csv(os.path.dirname(__file__)+ "/data/SuicidiosProyecto.csv", sep = ',')
X_train, X_test, y_train, y_test = train_test_split(comments_df['text'], comments_df['class'], test_size = 0.2, stratify = comments_df['class'], random_state = 1)


pipeline = Pipeline(
    [
        ('tfidf', TfidfVectorizer(tokenizer=tokenizer, stop_words = stop_words, lowercase = True)),
        ('model', LogisticRegression(random_state = 1, C=10, penalty='l2', solver='saga', max_iter=1000, n_jobs = -1))
    ]
)


print("Fitting the pipeline")
pipeline.fit(X_train, y_train)

y_train_logistic_regression = pipeline.predict(X_train)
y_test_logistic_regression = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_test_logistic_regression)}")
print(f"Precision: {precision_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
print(f"Recall: {recall_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
print(f"F1: {f1_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
print("\n"+classification_report(y_test, y_test_logistic_regression))

dump(pipeline, "./models/logistic_regression.joblib")