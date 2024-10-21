'''Program to train model for classification of data to Hogsworth Houses (4 houses). Output is file for weights in logistic regression'''

import sys
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression

def filter_features(df, feature_to_filter) -> list[str]:
    '''filter columns in df to be included in analysis'''
    result:list[str] = df.columns.tolist()
    def filter_feature(feature):
        if feature in feature_to_filter:
            return False
        else:
            return True
    return list(filter(filter_feature, result))

def main()-> None:
    '''Program input error checking and training model using Logistic Regression class'''
    if (len(sys.argv) != 2):
         raise ValueError("Incorrect No of arguments. Expected 1 argument which is training file")
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(f"File reading Error:{e}")
    # From previous exercise, exclude the following features:
    # 1) Defense Against the Dark Arts (since it is correlated with Astronomy)
    # 2) Birthday, Best Hand, Arithmancy, Care of Magical Creatures since they have homogenous score distribution between the 4 houses
    # model = LogisticRegression()
    y_Ravenclaw:np.ndarray = df["Hogwarts House"].apply(lambda x: 1 if x=="Ravenclaw" else 0).to_numpy()
    y_Slytherin:np.ndarray = df["Hogwarts House"].apply(lambda x: 1 if x=="Slytherin" else 0).to_numpy()
    y_Gryffindor:np.ndarray = df["Hogwarts House"].apply(lambda x: 1 if x=="Gryffindor" else 0).to_numpy()
    y_Hufflepuff:np.ndarray = df["Hogwarts House"].apply(lambda x: 1 if x=="Hufflepuff" else 0).to_numpy()
    features_to_drop:list[str] = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand", "Arithmancy", "Care of Magical Creatures", "Defense Against the Dark Arts"] 
    df = df.drop(features_to_drop, axis = 1)
    
    


if __name__ == "__main__":
    main()