import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self,x):
        self.W = np.random.randn(x.shape[1], 1)
        self.b = np.random.randn(1)
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self,X):
        Z = X.dot(self.W) + self.b
        A = self.sigmoid(Z)
        return A
    
    def log_loss(self,y, A):
        return 1/len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
    
    def predict(self,X):
        A = self.forward(X)
        return A >= 0.5  

    
    def compute_prob(self,X):
        A = self.forward(X)
        return A   
    
    def gradients(self,X, A, y):
        dW = 1/len(y) * np.dot(X.T, A - y)
        db = 1/len(y) * np.sum(A - y)
        return (dW, db)
    
    def backward(self,X, A, y, learning_rate):
        dW, db = self.gradients(X, A, y)
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db
        
    def fit(self,X,y,alpha = 0.001,n_iter = 1000):
        loss_history = []

        # Training
        for i in range(n_iter):
            A = self.forward(X)
            loss_history.append(self.log_loss(y, A))
            self.backward(X, A, y, learning_rate=0.1)
        plt.figure(figsize=(9, 6))
        plt.plot(loss_history)

def plot_boundary(X,y,L):
    resolution = 300
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x1 = np.linspace(xlim[0], xlim[1], resolution)
    x2 = np.linspace(ylim[0], ylim[1], resolution)
    X1, X2 = np.meshgrid(x1, x2)
    XX = np.vstack((X1.ravel(), X2.ravel())).T
    print(XX.shape)
    Z = L.predict(XX)
    Z = Z.reshape((resolution, resolution))
    
    ax.pcolormesh(X1, X2, Z, zorder=0, alpha=0.1)
    ax.contour(X1, X2, Z, colors='g')
    plt.show()