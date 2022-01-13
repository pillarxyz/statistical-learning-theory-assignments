import numpy as np


class Perceptron:
    
    # initialize perceptron class
    def __init__(self, eps=0.05, max_iters=1000):
        self.eps = eps
        self.best_params = None
        self.loss = None
        self.n_iters = None
        self.max_iters = max_iters

    # define activation function
    def activation(self, X):
        return np.where(X >= 0, 1, -1)

    # calculate loss function
    def compute_loss(self, X, y, w, b):
        n = X.shape[0]
        S = 0
        for i in range(n):
            linear_output = np.dot(X[i], w) + b
            if self.activation(linear_output) != y[i]:
                S += 1
        Ls = S / n
        return Ls

    # initialize parameters
    def init_params(self, dim):
        w_0 = np.zeros(dim)
        b_0 = 0
        return w_0, b_0

    # training loop
    def fit(self, X, y):
        #loop over exemplars and update weights
        n = X.shape[0]
        w, b = self.init_params(X.shape[1])
        loss = self.compute_loss(X, y, w, b)
        n_iters = 0
        losses = [loss]
        while loss >= self.eps:
            for i in range(n):
                linear_output = np.dot(X[i], w) + b
                if self.activation(linear_output) * y[i] < 0:
                    w = w + y[i] * X[i]
                    b = b + y[i]
            loss = self.compute_loss(X, y, w, b)
            losses.append(loss)
            n_iters = n_iters + 1
            if n_iters >= self.max_iters:
                break
        self.best_params = {
            'w': w,
            'b': b
        }
        self.loss = loss
        self.n_iters = n_iters
        return losses

    # prediction
    def predict(self, X):
        w = self.best_params.get('w')
        b = self.best_params.get('b')
        linear_output = np.dot(X, w) + b
        y_predicted = self.activation(linear_output)
        return y_predicted
    
    # scoring method
    def accuracy(self, y_true, y_pred):
        n = len(y_true)
        S = 0
        for i in range(n):
            if y_true[i] == y_pred[i]:
                S += 1
        acc = S / n
        return acc
        
        
# Adaline and Pocket inherit from the Perceptron class
# to avoid repeating code

class Adaline(Perceptron):
    
    # Adaline's loss function
    def compute_loss(self, X, y, w, b):
        n = X.shape[0]
        S = np.dot(y - (np.dot(w, X.T) + b), y - (np.dot(w, X.T) + b))
        Ls = S / n
        return Ls

    # training loop
    def fit(self, X, y):
        n = X.shape[0]
        w, b = self.init_params(X.shape[1])
        loss = self.compute_loss(X, y, w, b)
        n_iters = 0
        losses = [loss]
        while loss > self.eps:
            for i in range(n):
                linear_output = np.dot(X[i], w) + b
                e_i = y[i] - linear_output
                if e_i != 0:
                    w = w + 0.01 * e_i * X[i]
                    b = b + 0.01 * e_i 
            loss = self.compute_loss(X, y, w, b)
            losses.append(loss)
            n_iters = n_iters + 1
            if n_iters >= self.max_iters:
                break       
        self.best_params = {
            'w': w,
            'b': b
        }
        self.loss = loss
        self.n_iters = n_iters
        return losses
        

class Pocket(Perceptron):
    # training loop
    def fit(self, X, y):
        n = X.shape[0]
        w_s, b_s = self.init_params(X.shape[1])
        w, b = w_s, b_s
        loss_s = self.compute_loss(X, y, w_s, b_s)
        n_iters = 0
        losses = [loss_s]
        while loss_s > self.eps:
            for i in range(n):
                linear_output = np.dot(X[i], w) + b
                if self.activation(linear_output) * y[i] < 0:
                    w = w + y[i] * X[i]
                    b = b + y[i]
            loss = self.compute_loss(X, y, w, b)
            loss_s = self.compute_loss(X, y, w_s, b_s)
            if loss < loss_s:
                w_s = w
                b_s = b
                loss_s = loss
            losses.append(loss_s)
            n_iters = n_iters + 1
            if n_iters >= self.max_iters:
                break
        self.best_params = {
            'w': w_s,
            'b': b_s
        }
        self.loss = loss_s
        self.n_iters = n_iters
        return losses

    
def main():
    # Generate dataset
    
    print("=== Generating Dataset ===")
    X, y = datasets.make_moons(500, noise=0.25, random_state=1)
    y = np.where(y == 0, -1, 1)
    plt.title('Nonlinearly separable dataset')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.show()
    
    # Apply standard perceptron
    print("=== Perceptron ===")
    perceptron = Perceptron(max_iters=100)
    losses = perceptron.fit(X, y)
    hyperplane_params = perceptron.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    x = np.linspace(X[:, 0].min(), X[:, 0].max())
    hyperplane = -(w[0] / w[1]) * x - b / w[1]
    plt.title('Perceptron hyperplane')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.plot(x, hyperplane, c='red')
    plt.show()
    n_iters = perceptron.n_iters
    y_predicted = perceptron.predict(X)
    acc = perceptron.accuracy(y, y_predicted)
    print(f'The perceptron model accuracy reached {acc * 100}% in {n_iters} iterations')
    plt.title('Empirical error evolution of Perceptron')
    plt.plot(losses)
    plt.show()
    
    # Apply pocket
    print("=== Pocket ===")
    T_max = 100
    pocket = Pocket(max_iters=T_max)
    losses = pocket.fit(X, y)
    hyperplane_params = pocket.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    x = np.linspace(X[:, 0].min(), X[:, 0].max())
    hyperplane = -(w[0] / w[1]) * x - b / w[1]
    plt.title('Pocket hyperplane')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.plot(x, hyperplane, c='red')
    plt.show()
    n_iters = pocket.n_iters
    y_predicted = pocket.predict(X)
    acc = perceptron.accuracy(y, y_predicted)
    print(f'The perceptron model accuracy reached {acc * 100}% in {n_iters} iterations')
    plt.title('Empirical error evolution of Pocket')
    plt.plot(losses)
    plt.show()
    
    # Apply Adaline
    print("=== Adaline ===")
    T_max = 100
    adaline = Adaline(max_iters=T_max)
    losses = adaline.fit(X, y)
    hyperplane_params = adaline.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']
    x = np.linspace(X[:, 0].min(), X[:, 0].max())
    hyperplane = -(w[0] / w[1]) * x - b / w[1]
    plt.title('Adaline hyperplane')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
    plt.plot(x, hyperplane, c='red')
    plt.show()
    n_iters = adaline.n_iters
    y_predicted = adaline.predict(X)
    acc = perceptron.accuracy(y, y_predicted)
    print(f'The perceptron model accuracy reached {acc * 100}% in {n_iters} iterations')
    plt.title('Empirical error evolution of Adaline')
    plt.plot(losses)
    plt.show()
    
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    main()
