import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, learning_rate=0.001, max_iters=1000):
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
            
        if n_features == 1:
            w = w[:, 0][0]
        
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

    
def rmse_regression(y_true , y_predicted):
    sum_error = np.sum((y_predicted - y_true)**2) 
    return np.sqrt(sum_error/y_true.shape[0])

def main():
    print("==== Univariate Linear Regression ====")
    cars = pd.read_csv('data/cars.csv')
    X, y = cars[['speed']].values, cars[['dist']].values
    
    plt.scatter(X,y)
    plt.show()
    
    lr = LinearRegression(learning_rate=0.001)
    losses = lr.fit(X, y)
    hyperplane_params = lr.best_params
    
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    y_predicted = lr.predict(X)
    plt.title('Linear regression hyperplane')
    plt.scatter(X,y)
    plt.plot(X, y_predicted, c='red')
    plt.show()
    
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    rmse_score = rmse_regression(y, y_predicted)
    print(f'Linear regression model rmse is: {rmse_score}')
    
    plt.title('Empirical error evolution of Regression')
    plt.plot(losses)
    plt.show()
    
    print("==== Multivariate Linear Regression ====")
    data = pd.read_excel('data/pop.xlsx')
    
    X, y = data.drop(columns=['X1']), data['X1']
    X = StandardScaler().fit_transform(X)
    
    lr = LinearRegression(learning_rate=0.1, max_iters=50)
    losses = lr.fit(X, y)
    hyperplane_params = lr.best_params
    
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    print(f"the optimal parametres for our linear regression are w = {w}, b = {b}")
    
    plt.title('Empirical error evolution of Regression')
    plt.plot(losses)
    plt.show()
    
    y_predicted = lr.predict(X)
    rmse_score = rmse_regression(y, y_predicted)

    print(f'Linear regression model rmse is: {rmse_score}')
    
if __name__ == "__main__":
    main()