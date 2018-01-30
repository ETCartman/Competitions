"""
config.py
---
It will be used in all models.
---
"""


class Config(object):
    data_path = '../data/train.tsv'
    data_label = 'TYPE'
    data_text = 'ITEM_NAME'
    test_size = 0.2
    random_state = 42
    model_path = '../result/ml_model'
