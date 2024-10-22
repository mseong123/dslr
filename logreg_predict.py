'''Program to generate prediction file for classification in 4 Hogsworth houses
based on weights generated in training program.'''

import sys
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression

def main():
    '''Program argument error checking and create a prediction file for each house'''
    if len(sys.argv) != 3:
        raise ValueError("Incorrect No of arguments")
    try:
        # load the saved weights previously
        loaded = np.load("./weight.npz")
        df = pd.read_csv(f"./datasets/{sys.argv[1]}")
    except Exception as e:
        print(f"File reading Error:{e}")
    features_to_drop:list[str] = ["Index", "Hogwarts House", "First Name", \
                                  "Last Name", "Birthday", "Best Hand", "Arithmancy", \
                                    "Care of Magical Creatures", "Defense Against the Dark Arts"]
    df_processed:pd.DataFrame = df.drop(features_to_drop, axis = 1)
    model = LogisticRegression()
    for idx,_ in enumerate(loaded['weights']):
        model.weight = [loaded['weights'][idx]]
        model.bias = loaded['bias'][idx]
        print(model.weight)
        print(model.bias)
        result = model.predict(df_processed.to_numpy())
        print(result)
    
    
    


if __name__ == "__main__":
    main() #