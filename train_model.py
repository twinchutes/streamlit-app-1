# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# %%
def read_data():
    data = pd.read_csv('IMDB Dataset.csv')
    x = data[['review']].to_numpy()
    y = data[['sentiment']].to_numpy()
    return x, y

# %%
pipeline = Pipeline{[]}
vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
clf = MultinomialNB()

# %%
if __name__ == '__main__':
    x, y = read_data()
    


