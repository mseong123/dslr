'''Program to generate prediction file for classification in 4 Hogsworth houses
based on weights generated in training program.'''

import sys
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression

def handle_missing_data(df:pd.DataFrame) -> pd.DataFrame:
    '''convert NaN value into mean of the column'''
    for idx, _ in enumerate(df.columns):
        df[df.columns[idx]] = df[df.columns[idx]].fillna(value=np.mean(df[df.columns[idx]]))
    return df

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
    df_processed:pd.DataFrame = handle_missing_data(df_processed)
    houses:pd.DataFrame = pd.DataFrame({"Index":{}, "Hogwarts House":{}, })
    classification:list[str] = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    model = LogisticRegression()
    for idx,_ in enumerate(loaded['weights']):
        model.weight = loaded['weights'][idx]
        model.bias = loaded['bias'][idx]
        houses[classification[idx]] = model.predict(df_processed.to_numpy())

    houses["Index"] = df["Index"]
    houses["Hogwarts House"] = houses.apply(lambda row: \
        classification[np.argmax([row[classification[0]], row[classification[1]], \
        row[classification[2]], row[classification[3]]])], axis=1)
    houses = houses.drop(classification, axis=1)
    houses.to_csv("houses.csv", index=False)

if __name__ == "__main__":
    main()
