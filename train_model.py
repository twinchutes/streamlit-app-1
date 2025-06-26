import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def read_data():
    data = pd.read_csv('IMDB Dataset.csv')
    x = data['review']
    y = data['sentiment']
    return x, y

def train_model(x, y):
    vectorizer = TfidfVectorizer(stop_words="english")
    clf = MultinomialNB()
    pipe = Pipeline(steps=[
        ('vectorizer', vectorizer),
        ('clf', clf)
    ])
    return pipe.fit(x, y)

def dump_to_file(pipeline_object):
    joblib.dump(pipeline_object, 'sentiment_analysis.pkl')

if __name__ == '__main__':
    x, y = read_data()
    model = train_model(x, y)
    dump_to_file(model)