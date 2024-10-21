'''Logistic Regression class with methods and attributes to follow Scikit-learn prototype'''
import numpy as np

class LogisticRegression():
    '''Logistic Regression class'''
    def __init__(self, iterations:int = 1000, learning_rate:float = 0.001):
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def binary_cross_entropy_loss()->np.ndarray:

    def sigmoid(x:np.ndarray)->np.ndarray:
        return 1 / 1 + np.exp(-x)

    def fit(y:np.ndarray, X:np.ndarray)-> None:

        
    