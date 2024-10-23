'''Logistic Regression class with methods and attributes to follow Scikit-learn prototype'''
import numpy as np

class LogisticRegression():
    '''Logistic Regression class using one vs all (Sigmoid and binary cross entropy)'''
    def __init__(self, iterations:int = 1000, learning_rate:float = 0.1):
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._weight = np.array([])
        self._bias = np.array([])

    @property
    def weight(self) -> np.ndarray:
        '''return weight of instance'''
        return self._weight

    @property
    def bias(self) -> np.ndarray:
        '''return bias of instance'''
        return self._bias

    @weight.setter
    def weight(self, value:np.ndarray) -> None:
        '''setting weight of instance'''
        self._weight = value

    @bias.setter
    def bias(self, value:np.ndarray) -> None:
        '''setting bias of instance'''
        self._bias = value

    def binary_cross_entropy_loss(self, y:np.ndarray, y_predict:np.ndarray)->np.ndarray:
        '''calculate and return loss value'''
        epsilon = 1e-10
        # shape is (size of sample,)
        # add epsilon to prevent log(0) which is undefined
        return np.mean(-((y * np.log(y_predict + epsilon)) + \
                        ((1 - y) * (np.log(1 - y_predict + epsilon)))))

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
        gradient_weight:np.ndarray = np.dot(X.T, y_predict - y) / len(y_predict)
        gradient_bias:np.ndarray = np.mean(y_predict - y)
        return (gradient_weight, gradient_bias)

    def predict(self, X:np.ndarray) -> np.ndarray:
        '''return probability of prediction based on data'''
        X = self.normalize(X)
        z:float = np.dot(X, self._weight) + self._bias
        return self.sigmoid(z)

    def normalize(self, X:np.ndarray) -> np.ndarray:
        '''convert each value to normalised Z score with mean of 0 and std deviation of 1'''
        # have to normalize so z doesn't overflow sigmoid
        return ((X - np.mean(X,axis=0).reshape(1,-1)) / np.std(X,axis=0).reshape(1,-1))

    def fit(self, y:np.ndarray, X:np.ndarray)-> None:
        '''train model based on data'''
        # weights = vector of features
        self._weight:np.ndarray = np.zeros(X.shape[1])
        self._bias:np.ndarray = np.zeros(1)
        X = self.normalize(X)
        for i in range(self._iterations):
            y_predict:float = self.sigmoid(np.dot(X, self._weight) + self._bias)
            loss = self.binary_cross_entropy_loss(y, y_predict)
            if i % 100 == 0:
                print(f"Iteration: {i}, Loss value: {loss}")
            # GD Update step
            gradient_weight, gradient_bias = self.gradient(y, y_predict, X)
            self._weight -= self._learning_rate * gradient_weight
            self._bias -= self._learning_rate * gradient_bias
