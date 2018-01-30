import pandas as pd
from config import Config
from model.Data_shuffle import DataShuffle
from model.pipelines import data_label, data_text
from model import Model

"""
processing.py
---
load_data() will shuffle data,split data
,save text and label model and return X_train, y_train,
 X_test, y_test
"""


def load_data():
    model = Model.Model("result/ml_model/")
    print("[*] Loading data...")
    data = pd.read_csv('data/data.tsv', sep='\t')
    print("[+] Data Loaded!")
    print("[*] Shuffle data...")
    data = DataShuffle().fit_transform(data)
    test_len = int(len(data)*Config.test_size)
    train_set = data[test_len:]
    test_set = data[:test_len]
    print("[+] Shuffle completed!")
    print("[*] Data processing...")
    X_train = data_text.fit_transform(train_set)
    y_train = data_label.fit_transform(train_set)
    X_test = data_text.transform(test_set)
    y_test = data_label.transform(test_set)
    print("[+] Process completed!")
    model.save_model(data_text, "data_text.model")
    model.save_model(data_label, "data_label.model")

    return X_train, y_train, X_test, y_test

