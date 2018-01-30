import joblib
import config

"""
Model.py
---
It will save or load the model which you have trained. 
"""
config = config.Config


class Model(object):
    """Save Model or Load Model"""
    def __init__(self, path=config.model_path):
        self.path = path

    def save_model(self, model, name):
        print("[*] Saving model: {}".format(name))
        try:
            joblib.dump(model, self.path + name)
        except IOError:
            print("[!] Unable to save!")
            exit()
        else:
            print("[+] Model saved!")

    def load_model(self, name):
        print("[*] Loading Model: {}".format(name))
        model = None
        try:
            model = joblib.load(self.path + name)
        except IOError:
            print("[!] Unable to find model!")
            exit()
        else:
            print("[+] Model loaded!")
        return model
