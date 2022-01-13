import numpy as np
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# define function to map higher order polynomial features
def map_feature(X1, X2, degree):
    res = np.zeros(X1.shape[0])
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            res = np.column_stack((res, (X1 ** (i-j)) * (X2 ** j)))
    return res

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# draw the decision boundary
def plot_decision_boundary(w, b, degree, axes):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U,V = np.meshgrid(u,v)
    # convert U, V to vectors for calculating additional features
    # using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))
    
    X_poly = map_feature(U, V, degree)
    Z = X_poly.dot(w) + b
    
    # reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))
    
    cs = axes.contour(U,V,Z,levels=[0],cmap= "Greys_r")

    return cs

# create a non linear data
data = pd.read_csv('circles.csv')
data = data.sample(n=len(data), random_state=42).reset_index(drop=True)

X_ = np.array(data.iloc[:,:2])
y = np.array(data.iloc[:,2])

# apply non linear transformation

for degree in range(1, 7):
    print(f"==== Q = {degree} ====")
    X = map_feature(X_[:,0], X_[:,1], degree = degree)
    
    n_features = X.shape[1]
    print(f"the number of features is {n_features}")
    
    # split the data to train and test set
    X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.15, shuffle = True, random_state = 1)

    # initiate the model
    model = LogisticRegression()

    # train the model
    losses = model.fit(X_train, y_train)
    hyperplane_params = model.best_params
    w = hyperplane_params['w']
    b = hyperplane_params['b']

    # predict
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    # calculate the accuracy
    print(f"Train accuracy: {model.accuracy(y_train, y_train_pred)}")
    print(f"Test accuracy: {model.accuracy(y_test, y_pred)}")

    # get positive and negative samples for plotting
    pos = data.iloc[:,-1] == 1
    neg = data.iloc[:,-1] == 0

    # Visualize Data
    fig, axes = plt.subplots()
    axes.set_xlabel('Feature 1')
    axes.set_ylabel('Feature 2')

    axes.scatter(data.iloc[:,0], data.iloc[:,1],c=data.iloc[:,2])

    plot_decision_boundary(w, b, degree, axes)

    plt.show()