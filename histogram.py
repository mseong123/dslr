import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_feature(feature:str, df:pd.DataFrame, houses:list[str]) -> list:
    '''return a list of features Series based on houses'''
    houses:list[str] = ["Ravenclaw", "Slytherin","Hufflepuff", "Gryffindor"]
    print(len([df[df["Hogwarts House"] == house][feature].tolist() for house in houses][1]))
    return [df[df["Hogwarts House"] == house][feature].tolist() for house in houses]


    




def main() -> None:
    '''Create histograms of homogenous scores between the 4 houses for each feature and save them'''
    plt.close("all")
    df = pd.read_csv("./datasets/dataset_train.csv")
    houses:list[str] = ["Ravenclaw", "Slytherin","Hufflepuff", "Gryffindor"]
    color:list[str] = ["red", "green","yellow", "blue"]
    data = parse_feature("Arithmancy", df, houses)
    fig, ax = plt.subplots(layout='constrained')   
    for idx, house in enumerate(houses):
        ax.hist(data[idx], color=color[idx], label=house, stacked=True, alpha=0.5)

   
    ax.set_title("Arithmancy")
    plt.legend()
    plt.show()
    

    


if __name__ == "__main__":
    main()

