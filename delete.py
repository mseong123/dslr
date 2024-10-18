'''delete files and folders created during evaluation'''
import shutil
import os

def main():
    '''delete files and folders created during evaluation'''
    directory_path:str = "./scatterplot"
    shutil.rmtree(directory_path, ignore_errors=True)

if __name__ == "__main__":
    main()