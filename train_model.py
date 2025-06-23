# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# %%
data = pd.read_csv('IMDB Dataset.csv')
x = data[['review']].to_numpy()
y = data[['sentiment']].to_numpy()

# %%



