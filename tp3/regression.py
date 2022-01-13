import numpy as np
from scipy.misc import derivative as deriv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.best_params = None
        self.learning_rate = learning_rate
        self.loss = None
        self.max_iters = max_iters


    def compute_loss(self, X, y, w, b):
        n_samples = X.shape[0]
        y_predicted = np.dot(X, w) + b
        Ls = 1 / n_samples * np.sum(np.square(y_predicted - y))
        return Ls

    def init_params(self, n_features):
        w_0 = np.zeros(n_features)
        b_0 = 0
        return w_0, b_0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w, b = self.init_params(n_features)
        loss = self.compute_loss(X, y, w, b)
        losses = [loss]
        for _ in range(self.max_iters):
            y_predicted = np.dot(X, w) + b
            dw = 1 / n_samples * np.dot(X.T, y_predicted - y)
            db = 1 / n_samples * np.sum(y_predicted - y)
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
        y_predicted = np.dot(X, w) + b
        return y_predicted

# Univariate Polynomial Regression

class PolynomailRegression:
    def __init__(self, degree, learning_rate, iterations=10000):
        self.degree = degree
        self.learning_rate = learning_rate  
        self.iterations = iterations

	# function to transform X
    def transform(self, X) :
        # initialize X_transform
        X_transform = np.ones((self.m, 1))
        j = 0
        for j in range(self.degree + 1) :
            if j != 0 :
                x_pow = np.power(X, j)
                # append x_pow to X_transform
                X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis = 1)
        return X_transform

    # function to normalize X_transform
    def normalize(self, X) :
        X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis = 0)) / np.std(X[:, 1:], axis = 0)
        return X
        
    # model training
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.m, self.n = self.X.shape

        # weight initialization
        self.W = np.zeros(self.degree + 1)
        self.errors = []
        
        # transform X for polynomial h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
        X_transform = self.transform(self.X)
        
        # normalize X_transform
        X_normalize = self.normalize(X_transform)

                
        # gradient descent learning	
        for i in range(self.iterations):
            h = self.predict(self.X)
            error = h - self.Y
            self.errors.append(-np.sum(error))

            # update weights
            self.W = self.W - self.learning_rate * ( 1 / self.m ) * np.dot(X_normalize.T, error)
        
        return self

    # predict

    def predict(self, X):
        # transform X for polynomial h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
        X_transform = self.transform(X)
        X_normalize = self.normalize(X_transform)
        return np.dot(X_transform, self.W)

	
def main():
    data = pd.read_csv("data/temperature.csv")
    
    x, y = data[['temperature']].values, data['pressure'].values
    x = MinMaxScaler().fit_transform(x)
    
    print("=== Plotting Data ===")
    plt.plot(data["temperature"],data["pressure"])
    plt.xlabel("temperature")
    plt.ylabel("pressure")
    plt.show()
    
    print("=== Linear Regression ===")
    L = LinearRegression()
    losses = L.fit(x,y)
    
    print(f"the empirical error of Linear Regression is : {losses[-1]}")

    plt.title("Evolution of the empirical error of Linear Regression")
    plt.plot(losses)
    plt.show()
    
    plt.title("Linear Regression model")
    plt.scatter(x,y,c='b')
    plt.plot(x,L.predict(x),c='red')
    plt.show()
    
    
    print("=== Polynomial Regression ===")
    models = []
    for degree in range(2,5):
        print(f"=== {degree}th degree Regression ===")
        model = PolynomailRegression(degree = degree, learning_rate = 0.01)
        model.fit(x,y)
        models.append(model)
        print(f"the empirical error of order {degree} regression is : {model.errors[-1]}")
        plt.title(f"the empirical error of order {degree} regression")
        plt.plot(model.errors)
        plt.show()

    plt.figure(figsize=(10,5))
    plt.scatter(x,y,color="blue")

    plt.plot(x, models[0].predict(x), label="order 2")
    plt.plot(x, models[1].predict(x), label="order 3")
    plt.plot(x, models[2].predict(x), label="order 4")
    plt.title("Polynomial Regression model order 2,3,4")
    plt.xlabel("Temperature")
    plt.ylabel("Pressure")


    plt.legend()
    plt.show()
    
    e = 0.0000001
    err = 1
    iters = [i for i in range(100, 1000, 100)]
    lrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    errors = []
    max_iters = []
    for lr in lrs:
        for iterr in iters:
            best_model = PolynomailRegression(degree = 4, learning_rate = lr, iterations = iterr)
            best_model.fit(x, y)
            err = best_model.errors[-1]
            if err < e :
                errors.append(err)
                max_iters.append(iterr)
                break 

    df = pd.DataFrame({'learning rate' : lrs,
                       'emperical errors' : errors,
                       'max iterations' : max_iters,})
    
    print(df)
    
if __name__ == "__main__":
    main()

