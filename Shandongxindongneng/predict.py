from model import Model
import pandas as pd

"""
predict.py
---
It will predict the ITEM_NAME you given.
"""

model = Model.Model(path='result/ml_model/')

data_label = model.load_model('data_label.model')
data_text = model.load_model('data_text.model')
data_clf = model.load_model('data_clf.model')
cat_dict = model.load_model('cat_dict.model')

while True:
    name = input('Item Name: ')
    name = pd.DataFrame([name], columns=["ITEM_NAME"])
    X = data_text.transform(name)
    pred = data_clf.predict(X)
    print(cat_dict[pred[0]] + '--' + pred[0])
