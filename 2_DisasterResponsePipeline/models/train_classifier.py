# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from IPython import display
import warnings

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download(['wordnet', 'punkt', 'stopwords'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, BaseEstimator, TfidfTransformer
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score

import pickle

def load_data(database_filepath):
   
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse;", engine)
    X = df["message"].values
    category_names = df.iloc[:, 4:].columns.tolist()
    Y = df.iloc[:, 4:].values

    return X, Y, category_names


def tokenize(text):
    
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    result = []

    for token in tokens:
        if token in stop_words:
            result.append(lemmatizer.lemmatize(token))
    
    return result


def build_model():
   
    # initial basic model
    forest = RandomForestClassifier(random_state=42, n_jobs=4)

    # create pipeline
    pipeline = Pipeline([
        ("text_pipeline", Pipeline([
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer())
        ])),

        ("clf", MultiOutputClassifier(forest, n_jobs=4))
    ])

    # define parameters
    params = {
        "text_pipeline__vect__max_features": (5000, 10000),
        "clf__estimator__n_estimators": [50, 100, 150],
        "clf__estimator__criterion": ["gini", "entropy"],
        "clf__estimator__max_depth": [4, 6, 10],
    }

    # choose a method to build model
    print("Hint: GridSearchCV will takes more time!\n")
    chose_option = input("Choose the GridSearchCV(Yes or No): ").lower()
    while True:
        if chose_option in ["yes", "y"]:
            model = GridSearchCV(pipeline, param_grid=params, cv=3)
        elif chose_option in ["no", "n"]:
            model = pipeline
        else:
            print("Choose a validate option!")
            model = False
        
        if model:
            return model

def evaluate_model(model, X_test, Y_test, category_names):

    # predict the values
    y_pred = model.predict(X_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # display the report
        for index, column in enumerate(category_names):
            report = classification_report(
                Y_test[:, index], y_pred[:, index], labels=np.arange(1), target_names=[column], digits=2
            )
            print("{0:<25}\n  Precision:{1: =6.3f}   Recall:{2:< 5.3f}  F1 score:{3: 5.3f}".format(
                column, accuracy_score(Y_test[:, index], y_pred[:, index]),
                0, 1, 2
            ))


def save_model(model, model_filepath):
   
    # write the model object
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()