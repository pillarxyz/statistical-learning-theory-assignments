import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.best_params = None
        self.learning_rate = learning_rate
        self.loss = None
        self.max_iters = max_iters

    
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def activation(self, X):
        return np.where(X >= 0.5, 1, 0)

    def compute_loss(self, X, y, w, b):
        n_samples = X.shape[0]
        linear_output = np.dot(X, w) + b
        A = self.sigmoid(linear_output)
        Ls = -1 / n_samples * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        return Ls

    def init_params(self, n_features, initializer):
        if initializer == 'zero':
            w_0 = np.zeros(n_features)
            b_0 = 0
        if initializer == 'uniform':
            w_0 = np.random.uniform(0, 1, size = n_features)
            b_0 = np.random.uniform(0, 1)
        if initializer == 'gaussian':
            w_0 = np.random.normal(0.5, 0.5, size=n_features)
            b_0 = np.random.normal(0.5, 0.5)
        if initializer == 'logistic':
            w_0 = np.random.logistic(0.5, 0.15, size = n_features)
            b_0 = np.random.logistic(0.5, 0.15)
        return w_0, b_0

    def fit(self, X, y, initializer='uniform'):
        n_samples, n_features = X.shape
        w, b = self.init_params(n_features, initializer)
        loss = self.compute_loss(X, y, w, b)
        losses = [loss]
        for _ in range(self.max_iters):
            linear_output = np.dot(X, w) + b
            A = self.sigmoid(linear_output)
            dw = 1 / n_samples * np.dot(X.T, A - y)
            db = 1 / n_samples * np.sum(A - y)
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
            loss = self.compute_loss(X, y, w, b)
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
        linear_output = np.dot(X, w) + b
        A = self.activation(self.sigmoid(linear_output))
        return A

def accuracy_classification(y_true , y_predicted):
    n = len(y_true)
    S = 0
    for i in range(n):     
        if y_true[i] == y_predicted[i]:
            S += 1        
    acc = S / n
    return acc

def main():
    print("==== Logistic Regression ===")
    data = pd.read_csv('data/binary.csv')
    
    X, y = data.drop(columns=['admit','rank']), data['admit']
    X_scaled = StandardScaler().fit_transform(X)
    
    print("Paramaters initialized as 0")
    log_reg = LogisticRegression(learning_rate=0.01)
    losses = log_reg.fit(X_scaled, y, initializer='zero')
    
    hyperplane_params = log_reg.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    print(f'Logistic regresion model emperical loss is: {losses[-1]}')
    
    plt.title('Empirical error evolution of Logistic Regression')
    plt.plot(losses)
    plt.show()
    
    print("Paramaters initialized randomly (uniform)")
    random.seed(1234)
    log_reg = LogisticRegression(learning_rate=0.01)
    losses = log_reg.fit(X_scaled, y, initializer='uniform')
    
    hyperplane_params = log_reg.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    print(f'Logistic regresion model emperical loss is: {losses[-1]}')
    
    plt.title('Empirical error evolution of Logistic Regression')
    plt.plot(losses)
    plt.show()
    
    print("Paramaters initialized randomly (gaussian)")
    random.seed(1234)
    log_reg = LogisticRegression(learning_rate=0.01)
    losses = log_reg.fit(X_scaled, y, initializer='gaussian')
    
    hyperplane_params = log_reg.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    print(f'Logistic regresion model emperical loss is: {losses[-1]}')
    
    plt.title('Empirical error evolution of Logistic Regression')
    plt.plot(losses)
    plt.show()
    
    print("Paramaters initialized randomly (logistic)")
    random.seed(1234)
    log_reg = LogisticRegression(learning_rate=0.01)
    losses = log_reg.fit(X_scaled, y, initializer='logistic')
    
    hyperplane_params = log_reg.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    print(f'Logistic regresion model emperical loss is: {losses[-1]}')
    
    plt.title('Empirical error evolution of Logistic Regression')
    plt.plot(losses)
    plt.show()
    
    print("Adding rank as an extra feature")
    
    X, y = data.drop(columns=['admit']), data['admit']
    X_scaled = StandardScaler().fit_transform(X)
    
    print("Paramaters initialized as 0")
    log_reg = LogisticRegression(learning_rate=0.01)
    losses = log_reg.fit(X_scaled, y, initializer='zero')
    
    hyperplane_params = log_reg.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    print(f'Logistic regresion model emperical loss is: {losses[-1]}')
    
    plt.title('Empirical error evolution of Logistic Regression')
    plt.plot(losses)
    plt.show()
    
    print("Paramaters initialized randomly (uniform)")
    random.seed(1234)
    log_reg = LogisticRegression(learning_rate=0.01)
    losses = log_reg.fit(X_scaled, y, initializer='uniform')
    
    hyperplane_params = log_reg.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    print(f'Logistic regresion model emperical loss is: {losses[-1]}')
    
    plt.title('Empirical error evolution of Logistic Regression')
    plt.plot(losses)
    plt.show()
    
    print("Paramaters initialized randomly (gaussian)")
    random.seed(1234)
    log_reg = LogisticRegression(learning_rate=0.01)
    losses = log_reg.fit(X_scaled, y, initializer='gaussian')
    
    hyperplane_params = log_reg.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    print(f'Logistic regresion model emperical loss is: {losses[-1]}')
    
    plt.title('Empirical error evolution of Logistic Regression')
    plt.plot(losses)
    plt.show()
    
    print("Paramaters initialized randomly (logistic)")
    random.seed(1234)
    log_reg = LogisticRegression(learning_rate=0.01)
    losses = log_reg.fit(X_scaled, y, initializer='logistic')
    
    hyperplane_params = log_reg.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    print(f'Logistic regresion model emperical loss is: {losses[-1]}')
    
    plt.title('Empirical error evolution of Logistic Regression')
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()