'''Logistic Regression class with methods and attributes to follow Scikit-learn prototype'''
import numpy as np

class LogisticRegression():
    '''Logistic Regression class'''
    def __init__(self, iterations:int = 1000, learning_rate:float = 0.001):
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def binary_cross_entropy_loss(y:np.ndarray, y_predict:np.ndarray)->np.ndarray:
        '''calculate and return loss value'''
        # shape is (size of sample,)
        return -((y * np.log(y_predict)) + ((1 - y) * (np.log(1 - y_predict))))

    def sigmoid(z:np.ndarray)->np.ndarray:
        '''sigmoid value between 0 and 1'''
        # shape is (size of sample,)
        return 1 / (1 + np.exp(-z))
    
    def gradient(y, y_predict,X:np.ndarray) ->tuple[np.ndarray]:
        #dL/dy_predict = - ((y/y_predict) - (1-y/1-y_predict))
        # dy_predict/dz = y_predict * (1 - y_predict)
        # dz/dw 
        # gradient of loss w.r.t. dL/dw = (dL/dy_predict) x (dy_predict / dz) x (dz / dw)
        # simplify = ((y_predict - y) * X) / sample_size.
        # bias is the same except for dz/db = 1 hence (y_predict - y) / sample size
        weights:np.ndarray = 
        bias:np.ndarray = 

    
    def fit(y:np.ndarray, X:np.ndarray)-> None:
        '''train model based on data'''
        # weights = vector of features
        weights:np.ndarray = np.zeros(X.shape[1])
        bias:np.ndarray = np.zeros(1)
        for 
        
    