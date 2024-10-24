'''delete files and folders created during evaluation'''
import shutil
import os

def main():
    '''delete files and folders created during evaluation'''
    directory_path:str = "./scatterplot"
    shutil.rmtree(directory_path, ignore_errors=True)
    houses = "./houses.csv"
    weight = "./weight.npz"
    if os.path.exists(houses):
        os.remove(houses)
    if os.path.exists(weight):
        os.remove(weight)


if __name__ == "__main__":
    main()