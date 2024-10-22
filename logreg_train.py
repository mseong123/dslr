'''Program to train model for classification of data to Hogsworth Houses (4 houses). 
Output is file for weights in logistic regression'''

import sys
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression

def handle_missing_data(df:pd.DataFrame) -> pd.DataFrame:
    '''convert NaN value into mean of the column'''
    for i in range(len(df.columns)):
        df[df.columns[i]] = df[df.columns[i]].fillna(value=np.mean(df[df.columns[i]]))
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
    return { "y_ravenclaw" : y_ravenclaw, "y_slytherin":y_slytherin, "y_gryffindor":y_gryffindor, "y_hufflepuff":y_hufflepuff}

def normalize(X:np.ndarray) -> np.ndarray:
        '''convert each value to normalised Z score with mean of 0 and std deviation of 1'''
        return ((X - np.mean(X,axis=0).reshape(1,-1)) / np.std(X,axis=0).reshape(1,-1)) 

def main()-> None:
    '''Program input error checking and training model using Logistic Regression class'''
    if len(sys.argv) != 2:
         raise ValueError("Incorrect No of arguments. Expected 1 argument which is training file")
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(f"File reading Error:{e}")
    # From previous exercise, exclude the following features:
    # 1) Defense Against the Dark Arts (since it is correlated with Astronomy)
    # 2) Birthday, Best Hand, Arithmancy, Care of Magical Creatures since they have homogenous score distribution between the 4 houses
    # model = LogisticRegression()
    y_label:dict[str] = process_label(df)
    features_to_drop:list[str] = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand", "Arithmancy", "Care of Magical Creatures", "Defense Against the Dark Arts"] 
    df:pd.DataFrame = df.drop(features_to_drop, axis = 1)
    df:pd.DataFrame = handle_missing_data(df)
    model_ravenclaw = LogisticRegression()
    print("label", y_label["y_ravenclaw"])
    model_ravenclaw.fit(y_label["y_ravenclaw"], df.to_numpy())
    # print(df.loc[0:1].to_numpy())
    result=model_ravenclaw.predict(normalize(df.loc[0:].to_numpy()))
    np.savetxt("normalize.csv", normalize(df.loc[0:].to_numpy()), delimiter=",", fmt='%f')
    np.savetxt("weight.csv", model_ravenclaw.weight, delimiter=",", fmt='%f')
    np.savetxt("bias.csv", model_ravenclaw.bias, delimiter=",", fmt='%f')
    print(np.std(df['Astronomy']))
    np.savetxt("stddeviation.csv", [np.std(df['Astronomy'])], delimiter=",", fmt='%f')
    np.savetxt("mean.csv", [np.mean(df['Astronomy'])], delimiter=",", fmt='%f')
    # print(model_ravenclaw.weight)

    
    


if __name__ == "__main__":
    main()