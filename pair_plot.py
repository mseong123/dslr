'''
Program to create pairplot/scatterplot_matrix to show which Hogwarts feature are similar. 
'''

import textwrap
import matplotlib.pyplot as plt
import pandas as pd

def plot_scatterplot_matrix(df:pd.DataFrame, features:list[str]) -> None:
    '''render subplots of scatterplot_matrix based on data'''
    figure_size:tuple[int] = (14,10)
    _, ax = plt.subplots(len(features),len(features),layout='constrained', figsize=figure_size)
    for i, _ in enumerate(features):
        for j, _ in enumerate(features):
            ax[i, j].scatter(df[features[i]], df[features[j]], s=0.5)
            ax[i, j].set_yticks([])
            ax[i, j].set_xticks([])
            if j == 0:
                ax[i, j].set_ylabel('\n'.join(textwrap.wrap(df[features[i]].name, 10)), fontsize=7)
            if i == 0:
                ax[i, j].xaxis.set_label_position('top')
                ax[i, j].set_xlabel('\n'.join(textwrap.wrap(df[features[j]].name, 10)), fontsize=7)


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
    '''create scatterplot_matrix to show which Hogwarts feature are similar'''
    plt.close("all")
    df = pd.read_csv("./datasets/dataset_train.csv")
    feature_to_filter:list[str] = ["Index", "Hogwarts House", "First Name",\
                                    "Last Name", "Birthday", "Best Hand"]
    features:list[str] = filter_features(df, feature_to_filter)
    plot_scatterplot_matrix(df, features)
    plt.show()

if __name__ == "__main__":
    main()
