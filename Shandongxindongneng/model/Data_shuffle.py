from sklearn.base import BaseEstimator, TransformerMixin
from config import Config
import numpy as np

"""
Data_shuffle.py
---
The given data is ordered which will badly affect the
machine learning algorithm,so we need to shuffle the
data.
"""


class DataShuffle(BaseEstimator, TransformerMixin):
    def __init__(self, method='normal'):
        self.config = Config
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.method == 'normal':
            np.random.seed(Config.random_state)
            shuffle_index = np.random.permutation(len(X))
            return X.iloc[shuffle_index]
