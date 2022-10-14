import os
import string
import numpy as np
import pandas as pd
import logging
import sys

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from logger import define_log, StreamToLogger
from sklearn.metrics import classification_report

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

    # Función para tokenizar los tweets
    def tokenizer(text):
        tt = TweetTokenizer()
        tokens = tt.tokenize(text)
        tokens = [i for i in tokens if i not in string.punctuation]
        return tt.tokenize(text)

    # ### Loading tweets

    reviews_df = pd.read_csv(os.path.dirname(__file__)+ "/data/SuicidiosProyecto.csv", sep = ',')

    reviews_df.shape
    reviews_df.sample(5)
    reviews_df['class'].value_counts(dropna = False, normalize = True)

    # ### Filtrando los tweets que aún no han sido etiquetados

    reviews_labeled_df = reviews_df.loc[reviews_df['class'].notnull()]

    # Tweets a ser usados para entrenar y evaluar el modelo
    reviews_labeled_df.shape


    # Tweets descartados debido a que no se encuentran etiquetados
    reviews_df.loc[reviews_df['class'].isnull()].shape


    reviews_labeled_df['class'].value_counts(dropna = False, normalize = True)

    # ### Diviendo los tweets etiquetados en los conjuntos de entrenamiento y prueba

    # El parámetro 'stratify' es útil para asegurar que ambos conjuntos de datos queden aproximadamente balanceados
    # Cuando se cuenta con muy pocos datos etiquetados, por lo general se debe disponer de una menor cantidad de datos para evaluación
    X_train, X_test, y_train, y_test = train_test_split(reviews_labeled_df['text'], reviews_labeled_df['class'], test_size = 0.2, stratify = reviews_labeled_df['class'], random_state = 1)

    X_train.shape

    pd.Series(y_train).value_counts(normalize = True)

    X_test.shape
    pd.Series(y_test).value_counts(normalize = True)


    tfidf = TfidfVectorizer(tokenizer = tokenizer, stop_words = stop_words, lowercase = True)

    log.info("Applying fit transform over the TfidfVectorizer...")
    X_tfidf = tfidf.fit_transform(X_train)

    log.info(len(tfidf.vocabulary_))

    regression_params = [{
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['saga'],
        },
        {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['sag'],
        }]

    log.info("Training a Logistic Regression model...")
    logistic_regression = LogisticRegression(random_state = 1, max_iter=1000, n_jobs = -1)
    lr_search = GridSearchCV(logistic_regression, regression_params, cv = 3, n_jobs = -1, verbose = 1)
    lr_search.fit(X_tfidf, y_train)

    log.info("Best parameters: {}".format(lr_search.best_params_))
    log.info("Best score: {}".format(lr_search.best_score_))
    y_train_logistic_regression = lr_search.predict(X_tfidf)
    y_test_logistic_regression = lr_search.predict(tfidf.transform(X_test))
    log.info(f"Accuracy: {accuracy_score(y_test, y_test_logistic_regression)}")
    log.info(f"Precision: {precision_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
    log.info(f"Recall: {recall_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
    log.info(f"F1: {f1_score(y_test, y_test_logistic_regression, pos_label = 'non-suicide')}")
    log.info("\n"+classification_report(y_test, y_test_logistic_regression))

    forest_model = RandomForestClassifier(n_estimators=1000, random_state = 3, n_jobs=-1, verbose=1)
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

    mlp_model = MLPClassifier(
        max_iter=1000,
        hidden_layer_sizes=(20,10,10),
        random_state=1
    )

    param_space = {
        'activation' : ['tanh', 'relu'],
        'solver' : ['sgd', 'adam'],
        'alpha' : [0.001, 0.05],
        'learning_rate' : ['constant','adaptive']
    }

    log.info("Training Neural Network GridSearchCV...")
    clf = GridSearchCV(mlp_model, param_space, n_jobs=-1, cv=3, verbose=2)

    clf.fit(X_tfidf, y_train)

    log.info(clf.best_params_)

    preds_train = clf.predict(X_tfidf)
    preds_test = clf.predict(tfidf.transform(X_test))

    modelo_red_neuronal = pd.DataFrame([
        {
            'Train': accuracy_score(y_train, preds_train),
            'Test': accuracy_score(y_test, preds_test)
        },
        {
            'Train': precision_score(y_train, preds_train),
            'Test': precision_score(y_test, preds_test)
        },
        {
            'Train': recall_score(y_train, preds_train),
            'Test': recall_score(y_test, preds_test)
        },
        {
            'Train': f1_score(y_train, preds_train),
            'Test': f1_score(y_test, preds_test)
        }
    ], index = ['Accuracy', 'Precision', 'Recall', 'F1'])

    log.info(modelo_red_neuronal)
    log.info("\n"+classification_report(y_test, preds_test))