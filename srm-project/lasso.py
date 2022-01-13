import numpy as np
import pandas as pda
import matplotlib.pyplot as plt
import numpy as np


class LassoRegression:
    def __init__(self, learning_rate=0.001, max_iters=1000, alpha=1):
        self.best_params = None
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.loss = None
        self.max_iters = max_iters


    def compute_loss(self, X, y, w, b, alpha):
        n_samples = X.shape[0]
        y_predicted = np.dot(X, w) + b
        Ls = (np.sum(np.square(y_predicted - y)) + self.alpha * np.linalg.norm(w)) / (2 * n_samples)
        return Ls

    def init_params(self, n_features):
        w_0 = np.zeros(n_features)
        b_0 = 0
        return w_0, b_0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w, b = self.init_params(n_features)
        alpha = self.alpha
        loss = self.compute_loss(X, y, w, b, alpha)
        losses = [loss]
        for _ in range(self.max_iters):
            y_predicted = np.dot(X, w) + b
            dw = (2 / n_samples) * (np.dot(X.T, y_predicted - y) + self.alpha * np.sign(w))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
            loss = self.compute_loss(X, y, w, b, alpha)
            losses.append(loss)
        self.best_params = {
            'w': w,
            'b': b
        }
        self.loss = loss
        return losses

    def predict(self, X):
        w = self.best_params.get('w')
        b = self.best_params.get('b')
        y_predicted = np.dot(X, w) + b
        return y_predicted
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "alpha": self.alpha, "max_iters": self.max_iters}