'''program to evaluate performance of model to actual results'''

import pandas as pd
from sklearn.metrics import accuracy_score

def main()-> None:
    '''parsing csv file and comparing labels'''
    df_pred = pd.read_csv("./houses.csv")
    df_truth = pd.read_csv("./datasets/dataset_truth.csv")
    label_pred = df_pred["Hogwarts House"]
    label_truth = df_truth["Hogwarts House"]
    print(f"sci-kit accuracy score:{accuracy_score(label_truth, label_pred)}")


if __name__ == "__main__":
    main()