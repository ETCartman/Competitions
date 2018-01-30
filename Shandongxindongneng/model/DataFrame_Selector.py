from sklearn.base import BaseEstimator, TransformerMixin

"""
DataFrame_Selector.py
---
The given data has two columns, We need to process
each column in different way.
"""


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attr = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attr]
