# dslr | 42KL


![sigmoid curve](https://github.com/mseong123/dslr/blob/main/sigmoid.svg)

*Figure 1: Example sigmoid curve function, used as objective function in logistic regression.*

This project aims to introduce another core feature of Machine Learning domain which is Logistic Regression (also known as log loss regression). The project neccessitates the coding of the algorithms at the heart of the training process and is implemented using only Python, Numpy, Pandas and Matplotlib with no external ML libaries. 

Logistic Regression is a supervised learning and is used to classify data into separate classes based on features. This project incorporates element of sigmoid function, binary cross entropy function as a loss function and used gradient descent as optimization algorithm. 

Included in mandatory part:
 - Z score normalization of dataset
 - coding implementation of data science and statistical concepts
 - data cleaning (how to deal with missing data)
 - data visualisation (to check for collinearity between features using histogram, scatterplot and scatterplot matrix)
 - backpropagation of loss values to respective weights and biases using gradient descent

Included in BONUS part:
 - implementation of stochastic gradient descent
 - implementation of mini-batch gradient descent

The model achieved accuracy of 99% in all scenarios using test data as per Scikit-learn.metrics accuracy score. The classes and functions are coded using reference to Scikit-learn LogisticRegression class. 
Final results **125/100**

See [`Subject PDF`](https://github.com/mseong123/dslr/blob/main/en.subject.pdf) link.

## To install dependencies
```
pip install -r requirements.txt
```

## To run MANDATORY

### To train model
```
python logreg_train.py dataset_train.csv
```
This will train the regression model based on the `dataset_train.csv` and will show loss function value at every 100 iterations. Adjust hyperparameters learning rate and iteration in logreg_train.py to tune model's
performance. The above will output file `weight.npz` which will be used by model to predict classification output based on test dataset.

### To predict model
```
python logreg_predict.py dataset_test.csv weight.npz
```
This will output a file `houses.csv` which shows labelling of the classes in the test dataset. 

### To evaluate model
```
python evaluate.py
```
This will evaluate the labels generated in `houses.csv` against `dataset_truth.csv` using Scikit-learn accuracy_score method to evaluate precision of model (99%)

## To run BONUS

### To train model using SGD
```
python logreg_train.py dataset_train.csv SGD
```
To train model using stochastic gradient descent. Hyperparamater (epoch) can be adjusted.

### To train model using mini_batch
```
python logreg_train.py dataset_train.csv mini_batch
```
To train model using stochastic gradient descent. Hyperparamater (epoch and batch_size) can be adjusted.
Both models above can be evaluated using mandatory functions as shown above. Precision rate is 99% for both optimization methods.

