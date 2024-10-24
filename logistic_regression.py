'''Logistic Regression class with methods and attributes to follow Scikit-learn prototype'''
import numpy as np

class LogisticRegression():
    '''Logistic Regression class using one vs all (Sigmoid and binary cross entropy)'''
    def __init__(self, iterations:int = 1500, learning_rate:float = 0.01, epoch=5, batch_size=1):
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._epoch = epoch
        self._batch_size = batch_size
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

    def gradient(self, y:np.ndarray, y_predict:np.ndarray,X:np.ndarray,\
                  sgd:bool) ->tuple[np.ndarray]:
        '''calculate gradient of weight and bias'''
        #dL/dy_predict = - ((y/y_predict) - (1-y/1-y_predict))
        # dy_predict/dz = y_predict * (1 - y_predict)
        # dz/dw
        # gradient of loss w.r.t. dL/dw = (dL/dy_predict) x (dy_predict / dz) x (dz / dw)
        # simplify = ((y_predict - y) * X) / sample_size.
        # bias is the same except for dz/db = 1 hence (y_predict - y) / sample size
        if sgd is False:
            gradient_weight:np.ndarray = np.dot(X.T, y_predict - y) / len(y_predict)
        else:
            gradient_weight:np.ndarray = X * (y_predict - y)
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
 
    def shuffle(self, y:np.ndarray, X:np.ndarray, seed:int) -> None:
        '''shuffle label and data in place according to seed'''
        rng = np.random.default_rng(seed)
        indices = np.arange(len(y))
        rng.shuffle(indices)
        return (y[indices], X[indices])

    def fit(self, y:np.ndarray, X:np.ndarray, label:str)-> None:
        '''train model based on data'''
        # weights = vector of features
        print(f"\nStart training classifier for house {label}")
        self._weight:np.ndarray = np.zeros(X.shape[1])
        self._bias:np.ndarray = np.zeros(1)
        X = self.normalize(X)
        for i in range(self._iterations):
            y_predict:float = self.sigmoid(np.dot(X, self._weight) + self._bias)
            loss = self.binary_cross_entropy_loss(y, y_predict)
            if i % 100 == 0:
                print(f"Iteration: {i}, Loss value: {loss}")
            # GD Update step
            gradient_weight, gradient_bias = self.gradient(y, y_predict, X, False)
            self._weight -= self._learning_rate * gradient_weight
            self._bias -= self._learning_rate * gradient_bias

    def fit_sgd(self, y:np.ndarray, X:np.ndarray, label:str)-> None:
        '''train model based on data using pure SGD'''
        # weights = vector of features
        print(f"\nStart SGD training classifier for house {label}")
        self._weight:np.ndarray = np.zeros(X.shape[1])
        self._bias:np.ndarray = np.zeros(1)
        X = self.normalize(X)
        #shuffle initial dataset
        for i in range(self._epoch):
            # For each Epoch shuffle sample
            seed:int = np.random.randint(1,1000)
            (y_temp, X_temp)=self.shuffle(y, X, seed)
            if i in range(self._epoch):
                y_predict:float = self.sigmoid(np.dot(X_temp, self._weight) + self._bias)
                loss = self.binary_cross_entropy_loss(y_temp, y_predict)
                print(f"epoch: {i}, Loss value: {loss}")
            for j,_ in enumerate(X_temp):
                # Only 1 sample used in SGD update every iteration (up to entire dataset)
                y_predict:float = self.sigmoid(np.dot(_, self._weight) + self._bias)
                # SGD Update step
                gradient_weight, gradient_bias = self.gradient(y_temp[j], y_predict, _, True)
                self._weight -= self._learning_rate * gradient_weight
                self._bias -= self._learning_rate * gradient_bias

    def fit_mini_batch(self, y:np.ndarray, X:np.ndarray, label:str)-> None:
        '''train model based on data using mini-batch'''
        # weights = vector of features
        print(f"\nStart mini-batch training classifier for house {label}")
        self._weight:np.ndarray = np.zeros(X.shape[1])
        self._bias:np.ndarray = np.zeros(1)
        X = self.normalize(X)
        #shuffle initial dataset
        for i in range(self._epoch):
            # For each Epoch shuffle sample
            seed:int = np.random.randint(1,1000)
            (y_shuffle, X_shuffle)=self.shuffle(y, X, seed)
            if i in range(self._epoch):
                y_predict:float = self.sigmoid(np.dot(X_shuffle, self._weight) + self._bias)
                loss = self.binary_cross_entropy_loss(y_shuffle, y_predict)
                print(f"epoch: {i}, Loss value: {loss}")
            start_index:int = 0
            end_index:int = start_index + self._batch_size
            while end_index < len(X_shuffle):
                # mini batch SGD update every iteration (up to entire dataset for one epoch)
                X_batch = X_shuffle[start_index:end_index]
                y_batch = y_shuffle[start_index:end_index]
                y_predict:float = self.sigmoid(np.dot(X_batch, self._weight) + self._bias)
                # mini batch SGD Update step
                gradient_weight, gradient_bias = self.gradient(y_batch, y_predict, X_batch, False)
                self._weight -= self._learning_rate * gradient_weight
                self._bias -= self._learning_rate * gradient_bias
                start_index = end_index
                end_index = start_index + self._batch_size

