'''
Program to create scatterplot to show which Hogwarts feature are similar. 
'''
import os
import matplotlib.pyplot as plt
import pandas as pd

def parse_scatterplot_data(features:str, df:pd.DataFrame) -> list[list[pd.Series]]:
    '''return a list of dataframe pair feature data that can be used in scatterplot.
      ie 14 features, shape (14,[pd.Series] )'''
    result:list[list[pd.Series]] = []
    for idx, _ in enumerate(features):
        inner_result:list[pd.Series] = []
        inner_result.append(df[features[idx]])
        for j in range(idx+1, len(features)):
            inner_result.append(df[features[j]])
        result.append(inner_result)
    return result

def plot_scatterplot(data:list[list[pd.Series]]) -> None:
    '''render subplots of scatterplot based on data'''
    figure_size:tuple[int] = (14,10)
    try:
        os.mkdir("scatterplot/")
    except FileExistsError:
        pass
    for _, scatterplot in enumerate(data):
        fig, ax = plt.subplots(3,5,layout='constrained', figsize=figure_size)
        ax = ax.flatten()
        for i in range(1, len(scatterplot)):
            ax[i - 1].scatter(scatterplot[0], scatterplot[i])
            ax[i - 1].set_xlabel(scatterplot[0].name)
            ax[i - 1].set_ylabel(scatterplot[i].name)
        fig.savefig(f"scatterplot/{scatterplot[0].name}.png", format="png")

def filter_features(df, feature_to_filter) -> list[str]:
    '''filter columns in df to be included in analysis'''
    result:list[str] = df.columns.tolist()
    def filter_feature(feature):
        if feature in feature_to_filter:
            return False
        else:
            return True
    return list(filter(filter_feature, result))

def main() -> None:
    '''create scatterplot to show which Hogwarts feature are similar'''
    plt.close("all")
    df = pd.read_csv("./datasets/dataset_train.csv")
    feature_to_filter:list[str] = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday"]
    features:list[str] = filter_features(df, feature_to_filter)
    scatterplot_data:list[list[pd.Series]] = parse_scatterplot_data(features, df)
    plot_scatterplot(scatterplot_data)

if __name__ == "__main__":
    main()
