'''Logistic Regression class with methods and attributes to follow Scikit-learn prototype'''
import numpy as np

class LogisticRegression():
    '''Logistic Regression class using one vs all (Sigmoid and binary cross entropy)'''
    def __init__(self, iterations:int = 1000, learning_rate:float = 0.001):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weight = np.array([])
        self.bias = np.array([])
 
    def binary_cross_entropy_loss(self, y:np.ndarray, y_predict:np.ndarray)->np.ndarray:
        '''calculate and return loss value'''
        # shape is (size of sample,)
        return -((y * np.log(y_predict)) + ((1 - y) * (np.log(1 - y_predict))))

    def sigmoid(self, z:np.ndarray)->np.ndarray:
        '''sigmoid value between 0 and 1'''
        # shape is (size of sample,)
        return 1 / (1 + np.exp(-z))

    def gradient(self, y:np.ndarray, y_predict:np.ndarray,X:np.ndarray) ->tuple[np.ndarray]:
        '''calculate gradient of weight and bias'''
        #dL/dy_predict = - ((y/y_predict) - (1-y/1-y_predict))
        # dy_predict/dz = y_predict * (1 - y_predict)
        # dz/dw 
        # gradient of loss w.r.t. dL/dw = (dL/dy_predict) x (dy_predict / dz) x (dz / dw)
        # simplify = ((y_predict - y) * X) / sample_size.
        # bias is the same except for dz/db = 1 hence (y_predict - y) / sample size
        weight:np.ndarray = np.dot(y_predict - y, X.T) / len(y_predict)
        bias:np.ndarray = np.mean(y_predict - y)
        return (weight, bias)

    def predict(self, X:np.ndarray) -> np.ndarray:
        '''return probability of '''
        z:float = np.dot(X, self.weight) + self.bias
        return self.sigmoid(z)
 
    def fit(self, y:np.ndarray, X:np.ndarray)-> None:
        '''train model based on data'''
        # weights = vector of features
        self.weight:np.ndarray = np.zeros(X.shape[1])
        self.bias:np.ndarray = np.zeros(1)
        for i in range(self.iterations):
            y_predict = self.predict(X)
            loss = self.binary_cross_entropy_loss(y, y_predict)
            if i % 100 == 0:
                print(f"Iteration: {i}, Loss value: {loss}")
            # GD Update step
            gradient_weight, gradient_bias = self.gradient(y, y_predict, X)
            self.weight += self.weight - (self.learning_rate * gradient_weight)
            self.bias += self.bias - (self.learning_rate * gradient_bias)
