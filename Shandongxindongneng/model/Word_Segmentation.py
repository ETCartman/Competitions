import pandas as pd
import jieba
import config
from sklearn.base import BaseEstimator, TransformerMixin

"""
Word_Segmentation.py
---
This model will return word segmentation of 
ITEM_NAME.
"""


class WordSeg(BaseEstimator, TransformerMixin):
    def __init__(self, cut_all=False):
        self.config = config.Config
        self.cut_all = cut_all

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.Series([" ".join(jieba.cut(i, cut_all=self.cut_all)) for i in X])


if __name__ == "__main__":
    ws = WordSeg()
    data = pd.read_csv('../data/data.tsv', sep='\t')
    data['ITEM_NAME'] = ws.fit_transform(data['ITEM_NAME'])
    print(data)
