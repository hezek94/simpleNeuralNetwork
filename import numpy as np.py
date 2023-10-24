import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class neural_net(object):
    def __init__(self):
        self.inputLayers = 784
        self.outputlayers = 10
        self.hiddenlayers = 10
        self.W1 = np.random.rand(self.hiddenlayers, self.inputLayers) - 0.5
        self.b1 = np.random.rand(self.hiddenlayers, 1) - 0.5
        self.W2 = np.random.rand(self.outputlayers, self.hiddenlayers) - 0.5
        self.b2 = np.random.rand(self.outputlayers, 1) - 0.5
        self.m = 42000
        self.c = 0
        
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
        #return np.maximum(Z, 0)
    def forward_Prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.sigm_p(Z2)
        if self.c == 3:
            return A2
        return Z1, A1, Z2, A2

    def sigm_p(self, Z):
        a = np.exp(-Z)
        b = (1 + np.exp(-Z))
        return a / b**2

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    def Rel_U(self, Z):
        return Z>0

    def back_prop(self, X, Y):
        Y_n = self.one_hot(Y)
        Z1, A1, Z2, A2 = self.forward_Prop(X)
        dZ2 = A2 - Y_n
        dW2 = (1 / self.m) * dZ2.dot(A1.T)
        db2 = (1 / self.m) * np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * self.Rel_U(Z1)
        dW1 = (1 / self.m) * dZ1.dot(X.T)
        db1 = (1 / self.m) * np.sum(dZ1)

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2
        return self.W1, self.b1, self.W2, self.b2
        

    def get_predictions(self, A2):
        return np.argmax(A2, axis=0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y, alpha, iterations):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_Prop(X)
            dW1, db1, dW2, db2 = self.back_prop(X, Y)
            W1, b1, W2, b2 = self.update_params(dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(A2)
                print(self.get_accuracy(predictions, Y))
        return W1, b1, W2, b2

# Example usage:
# Define X and Y appropriately and call the gradient_descent method with your data.
data = pd.read_csv('D:/16187/Documents/train.csv')
data = np.array(data)
m, n = data.shape
print(m)
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
n_n=neural_net()
W1, b1, W2, b2 = n_n.gradient_descent(X_train, Y_train, 0.10, 500)