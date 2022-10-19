import os
from os import listdir
from os.path import isfile, join
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logger import define_log, StreamToLogger
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import logging

from joblib import dump

if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = "ignore:UserWarning"

    log = logging.getLogger('NLP Project')
    pd.set_option('display.max_colwidth', None)

    handler = define_log()
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    # Descargando las stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Función para tokenizar los comentarios
    def tokenizer(text):
        tt = TweetTokenizer()
        tokens = tt.tokenize(text)
        return tokens

    # ### Loading comentarios

    comments_df = pd.read_csv(os.path.dirname(__file__)+ "/data/SuicidiosProyecto.csv", sep = ',')

    comments_df.shape
    comments_df.sample(5)
    comments_df['class'].value_counts(dropna = False, normalize = True)

    # ### Filtrando los comentarios que aún no han sido etiquetados

    comments_labeled_df = comments_df.loc[comments_df['class'].notnull()]

    # comentarios a ser usados para entrenar y evaluar el modelo
    comments_labeled_df.shape


    # comentarios descartados debido a que no se encuentran etiquetados
    comments_df.loc[comments_df['class'].isnull()].shape


    comments_labeled_df['class'].value_counts(dropna = False, normalize = True)

    # ### Diviendo los comentarios etiquetados en los conjuntos de entrenamiento y prueba

    # El parámetro 'stratify' es útil para asegurar que ambos conjuntos de datos queden aproximadamente balanceados
    X_train, X_test, y_train, y_test = train_test_split(comments_labeled_df['text'], comments_labeled_df['class'], test_size = 0.2, stratify = comments_labeled_df['class'], random_state = 1)

    X_train.shape

    pd.Series(y_train).value_counts(normalize = True)

    X_test.shape
    pd.Series(y_test).value_counts(normalize = True)

    tfidf = TfidfVectorizer(tokenizer = tokenizer, stop_words = stop_words, lowercase = True)

    log.info("Applying fit transform over the TfidfVectorizer...")
    X_tfidf = tfidf.fit_transform(X_train)

    if not os.path.exists("./models/logistic_regression.joblib"):
        log.info("Training a Logistic Regression model...")
        lr_search = LogisticRegression(random_state = 1, C=10, penalty='l2', solver='saga', max_iter=1000, n_jobs = -1)
        lr_search.fit(X_tfidf, y_train)
        y_train_logistic_regression = lr_search.predict(X_tfidf)
        y_test_logistic_regression = lr_search.predict(tfidf.transform(X_test))
        log.info(f"Accuracy: {accuracy_score(y_test, y_test_logistic_regression)}")
        log.info(f"Precision: {precision_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
        log.info(f"Recall: {recall_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
        log.info(f"F1: {f1_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
        log.info("\n"+classification_report(y_test, y_test_logistic_regression))
        dump(lr_search, os.path.dirname(__file__) + "/models/logistic_regression.joblib")

    if not os.path.exists(os.path.dirname(__file__) + "/models/random_forest.joblib"):
        forest_model = RandomForestClassifier(n_estimators=1000, random_state = 3, n_jobs=-1, verbose=1)
        forest_params = {
            "n_estimators": [10, 100, 1000],
            'max_features': ['sqrt', 'log2'],
            'criterion' : ['gini', 'entropy']
        }
        log.info("Training the RandomForestClassifier...")
        forest_model.fit(X_tfidf, y_train)
        log.info(f"\n{pd.Series(forest_model.feature_importances_, index = tfidf.vocabulary_).sort_values().tail(20)}")
        forest_estimators = forest_model.estimators_
        log.info(f'Number of trees: {len(forest_estimators)}')
        log.info(f'Trees depth (mean): {np.mean([tree.get_depth() for tree in forest_estimators])}')
        y_train_tfidf_predict = forest_model.predict(X_tfidf)
        y_test_tfidf_predict = forest_model.predict(tfidf.transform(X_test))

        log.info(f"Accuracy: {accuracy_score(y_test, y_test_tfidf_predict)}")
        log.info(f"Precision: {precision_score(y_test, y_test_tfidf_predict, pos_label = 'non-suicide')}")
        log.info(f"Recall: {recall_score(y_test, y_test_tfidf_predict, pos_label = 'non-suicide')}")
        log.info(f"F1: {f1_score(y_test, y_test_tfidf_predict, pos_label = 'non-suicide')}")

        log.info("\n"+classification_report(y_test, y_test_tfidf_predict))
        dump(forest_model, os.path.dirname(__file__) + "/models/random_forest.joblib")

    if not os.path.exists("./models/neural_network.joblib"):
        clf = MLPClassifier(
            activation='relu',
            alpha=0.05,
            learning_rate='constant',
            solver='adam',
            max_iter=1000,
            hidden_layer_sizes=(20,10,10),
            random_state=1
        )

        log.info("Training Neural Network Classifier...")
        clf.fit(X_tfidf, y_train)

        preds_train = clf.predict(X_tfidf)
        preds_test = clf.predict(tfidf.transform(X_test))

        log.info(f"Accuracy: {accuracy_score(y_test, preds_test)}")
        log.info(f"Precision: {precision_score(y_test, preds_test, pos_label = 'non-suicide')}")
        log.info(f"Recall: {recall_score(y_test, preds_test, pos_label = 'non-suicide')}")
        log.info(f"F1: {f1_score(y_test, preds_test, pos_label = 'non-suicide')}")
        log.info("\n"+classification_report(y_test, preds_test))
        dump(clf, os.path.dirname(__file__) + "/models/neural_network.joblib")