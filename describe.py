'''Statistical analysis of dataset'''

import sys
import pandas as pd
import numpy as np

def count(result, data):
	'''count'''
	for column in result.columns:
		result.loc["Count",column] = data[column].notna().sum()

def mean(result, data):
	'''mean'''
	for column in result.columns:
		result.loc["Mean",column] = data[column].sum() / result.loc["Count", column]

def std(result, data):
	'''std deviation'''
	for column in result.columns:
		result.loc["Std",column] = np.sqrt(np.sum((data[column] - result.loc["Mean", column])**2) / result.loc["Count", column])

def min(result, data):
    '''min'''
    for column in result.columns:
        result.loc["Min", column] = data[column].sort_values().iloc[0]

def first_quartile(result, data):
	'''first_quatile'''
	for column in result.columns:
		result.loc["25%", column] = data[column].sort_values().iloc[round(result.loc["Count", column] * 0.25)]

def second_quartile(result, data):
    '''median'''
    for column in result.columns:
        result.loc["50%", column] = data[column].sort_values().iloc[round(result.loc["Count", column] * 0.5)]

def third_quartile(result, data):
	'''third quartile'''
	for column in result.columns:
		result.loc["75%", column] = data[column].sort_values().iloc[round(result.loc["Count", column] * 0.75)]

def max(result, data):
	'''max'''
	for column in result.columns:
		result.loc["Max", column] = data[column].sort_values().iloc[result.loc["Count",column] - 1]

def range(result, data):
	'''range'''
	for column in result.columns:
		result.loc["Range", column] = result.loc["Max",column] - result.loc["Min", column]

def IQR(result, data):
	'''Interquatile range'''
	for column in result.columns:
		result.loc["IQR", column] = result.loc["75%",column] - result.loc["25%", column]

def var(result, data):
	'''Interquatile range'''
	for column in result.columns:
		result.loc["Variance", column] = result.loc["Std",column]**2

def main():
	'''list describe() statistics include additional fields for bonus'''
	if len(sys.argv) != 2:
		print("Incorrect number of parameters. Program terminating")
		return(0)
	try:
		data = pd.read_csv(f"./datasets/{sys.argv[1]}")
	except Exception:
		print("Cannot parse input file. Program terminating")
		return(0)
	else:
		result = pd.DataFrame(index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"], columns = data.select_dtypes(exclude=[object]).drop(labels=["Index"], axis=1).columns)
		if "Hogwarts House" in result.columns:
			result = result.drop(labels="Hogwarts House", axis =1)
		count(result, data)
		mean(result, data)
		std(result,data)
		min(result, data)
		first_quartile(result,data)
		second_quartile(result,data)
		third_quartile(result,data)
		max(result,data)
		print("Mandatory fields\n")
		print(result)
		print("\n")
		bonus = pd.DataFrame(index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Range", "IQR", "Variance"], columns = data.select_dtypes(exclude=[object]).drop(labels=["Index"], axis=1).columns)
		if "Hogwarts House" in bonus.columns:
			bonus = bonus.drop(labels="Hogwarts House", axis =1)
		count(bonus, data)
		mean(bonus, data)
		std(bonus,data)
		min(bonus, data)
		first_quartile(bonus,data)
		second_quartile(bonus,data)
		third_quartile(bonus,data)
		max(bonus,data)
		range(bonus,data)
		IQR(bonus, data)
		var(bonus, data)
		
		print("Bonus Fields")
		print(bonus)



if __name__ == "__main__":
	main()