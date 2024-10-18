'''
Program to create histogram to show which Hogwarts feature has a homogeneous score 
distribution between all four houses.
'''
import matplotlib.pyplot as plt
import pandas as pd

def parse_feature(feature:str, df:pd.DataFrame, houses:list[str]) -> list:
    '''return a list of features Series based on houses'''
    return [df[df["Hogwarts House"] == house][feature].tolist() for house in houses]

def filter_features(df) -> list[str]:
    '''filter columns in df to be included in analysis'''
    result:list[str] = df.columns.tolist()
    def filter_feature(feature):
        feature_to_filter:list[str] = ["Index", "Hogwarts House", "First Name", "Last Name"]
        if feature in feature_to_filter:
            return False
        else:
            return True
    return list(filter(filter_feature, result))

def plot_histogram(data:list, houses:list[str], feature:str, \
                   ax, feature_idx:int) -> None:
    '''render subplots of histogram based on data'''
    color:list[str] = ["red", "green","yellow", "blue"]
    for idx, house in enumerate(houses):
        if feature == "Birthday":
            date_series = pd.Series(data[idx])
            dates = pd.to_datetime(date_series)
            ax[feature_idx].hist(dates, color=color[idx], label=house, stacked=True, alpha=0.5)
        else:
            ax[feature_idx].hist(data[idx], color=color[idx], label=house, stacked=True, alpha=0.5)
        ax[feature_idx].set_title(feature)



def main() -> None:
    '''Create histograms of homogenous scores between the 4 houses for each feature and save them'''
    plt.close("all")
    df = pd.read_csv("./datasets/dataset_train.csv")
    houses:list[str] = ["Ravenclaw", "Slytherin","Hufflepuff", "Gryffindor"]
    features:list[str] = filter_features(df)
    figure_size:tuple[int] = (14,10)

    _, ax = plt.subplots(3,5,layout='constrained', figsize=figure_size)
    ax = ax.flatten()
    for idx, feature in enumerate(features):
        data:list = parse_feature(feature, df, houses)
        plot_histogram(data,houses,feature, ax, idx)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()