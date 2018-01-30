from sklearn.pipeline import Pipeline
from config import Config
from DataFrame_Selector import DataFrameSelector
from Data_shuffle import DataShuffle
from Word_Segmentation import WordSeg
from Lable_Split import LabelSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

"""
pipelines.py
---
This model uses Pipeline to integarate all the models
data_text: process ITEM_NAME
data_lable: porcess Label
data_clf: LinearSVC Classifier
"""

data_text = Pipeline([
    ('selector', DataFrameSelector(Config.data_text)),
    ('shuffle', DataShuffle()),
    ('wordseg', WordSeg()),
])


data_label = Pipeline([
    ('selector', DataFrameSelector(Config.data_label)),
    ('shuffle', DataShuffle()),
    ('labelSplit', LabelSplit()),
])

data_clf = Pipeline([
    ('countvector', CountVectorizer(min_df=1, ngram_range=(1, 2),
                                    token_pattern='(?u)\\b\\w+\\b')),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC())),
])
