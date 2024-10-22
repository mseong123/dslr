'''Program to train model for classification of data to Hogsworth Houses (4 houses). 
Output is file for weights in logistic regression'''

import sys
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression

def handle_missing_data(df:pd.DataFrame) -> pd.DataFrame:
    '''convert NaN value into mean of the column'''
    for idx, _ in enumerate(df.columns):
        df[df.columns[idx]] = df[df.columns[idx]].fillna(value=np.mean(df[df.columns[idx]]))
    return df

def process_label(df:pd.DataFrame) -> dict[str]:
    """transform each classes to one vs all label """
    y_ravenclaw:np.ndarray = df["Hogwarts House"].\
        apply(lambda x:1 if x=="Ravenclaw" else 0).to_numpy()
    y_slytherin:np.ndarray = df["Hogwarts House"].\
        apply(lambda x:1 if x=="Slytherin" else 0).to_numpy()
    y_gryffindor:np.ndarray = df["Hogwarts House"].\
        apply(lambda x:1 if x=="Gryffindor" else 0).to_numpy()
    y_hufflepuff:np.ndarray = df["Hogwarts House"].\
        apply(lambda x:1 if x=="Hufflepuff" else 0).to_numpy()
    return { "y_ravenclaw" : y_ravenclaw, "y_slytherin":y_slytherin, \
            "y_gryffindor":y_gryffindor, "y_hufflepuff":y_hufflepuff}

def main()-> None:
    '''Progra error checking and training model using Logistic Regression class'''
    if len(sys.argv) != 2 or sys.argv[1] != "dataset_train.csv":
        raise ValueError("Incorrect No of arguments or wrong file name")
    try:
        df = pd.read_csv(f"./datasets/{sys.argv[1]}")
    except Exception as e:
        print(f"File reading Error:{e}")
    # From previous exercise, exclude the following features:
    # 1) Defense Against the Dark Arts (since it is correlated with Astronomy)
    # 2) Birthday, Best Hand, Arithmancy, Care of Magical Creatures since they
    # have homogenous score distribution between the 4 houses
    y_label:dict[str] = process_label(df)
    features_to_drop:list[str] = ["Index", "Hogwarts House", "First Name", \
                                  "Last Name", "Birthday", "Best Hand", "Arithmancy", \
                                    "Care of Magical Creatures", "Defense Against the Dark Arts"] 
    df:pd.DataFrame = df.drop(features_to_drop, axis = 1)
    df:pd.DataFrame = handle_missing_data(df)
    #train a model classifier for each class (4 in total)
    model_ravenclaw = LogisticRegression()
    model_ravenclaw.fit(y_label["y_ravenclaw"], df.to_numpy())
    model_slytherin = LogisticRegression()
    model_slytherin.fit(y_label["y_slytherin"], df.to_numpy())
    model_gryffindor = LogisticRegression()
    model_gryffindor.fit(y_label["y_gryffindor"], df.to_numpy())
    model_hufflepuff = LogisticRegression()
    model_hufflepuff.fit(y_label["y_hufflepuff"], df.to_numpy())
    #save model
    np.savez("weight", weights=[model_ravenclaw.weight, model_slytherin.weight, \
                           model_gryffindor.weight, model_hufflepuff.weight], \
                            bias = [model_ravenclaw.bias, model_slytherin.bias, \
                           model_gryffindor.bias, model_hufflepuff.bias] )

if __name__ == "__main__":
    main()
