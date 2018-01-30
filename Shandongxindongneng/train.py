from model.pipelines import data_clf
from model import Model
from processing import load_data

"""
train.py
---
Train and save model.
"""
__author__ = 'Hanjun Liu'
__email__ = 'liuhanjun369@gmail.com'


Model = Model.Model(path='result/ml_model/')
X_train, y_train, X_test, y_test = load_data()

print("[*] Training model...")
data_clf.fit(X_train, y_train)
print("[+] Training completed!")
print("[*] Testing Model...")
print(data_clf.score(X_test, y_test))
print("[+] Testing over!")

Model.save_model(data_clf, 'data_clf.model')
