

import numpy as np
from tensorflow.keras.models import load_model

class EMDAM():
    def __init__(self, model_paths):
        models = []
        for model_path in model_paths:
            model = load_model(model_path)
            models.append(model)
            self.input_size = (model.input.shape[1], model.input.shape[2])
        self.models = models

    def predict(self, x):
        y_preds = []
        for model in self.models:
            y_pred = model.predict(x)
            y_preds.append(y_pred)
        y_mean_pred = np.mean(y_preds, axis=0)
        return y_mean_pred

class SingleModel():
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.input_size = (self.model.input.shape[1], self.model.input.shape[2])

    def predict(self, x):
        return self.model.predict(x)