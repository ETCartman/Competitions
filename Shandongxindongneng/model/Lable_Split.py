from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from Model import Model

"""
Label_Split.py
---
The given data has three categories,We only need
the last one,This model will return the last label
and make and save a dictionary in {Label3: "Label1--Label2"} 
format.
"""


class LabelSplit(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = Model('result/ml_model/')
        self.label_dict = {}

    def get_dict(self, data):
        for t in data:
            type_list = t.split('--')
            self.label_dict[type_list[-1]] = type_list[0] + '--' + type_list[1]
        self.model.save_model(self.label_dict,'cat_dict.model')

    def fit(self, X, y=None):
        self.get_dict(X)
        return self

    def transform(self, X, y=None):
        return pd.Series([i.split('--')[-1] for i in X])


if __name__ == '__main__':
    ls = LabelSplit()
    data = pd.read_csv('../data/data.tsv', sep='\t')
    data['TYPE'] = ls.fit_transform(data['TYPE'])
    print(data)
