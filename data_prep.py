import numpy as np
import pandas as pd

# def fill_missing():
#     pass


# missing data statistics: 
def find_missing(df):
    if df.isnull().values.any():
        missing_value_counts = df.isnull().sum()
        missing_value_rates = (df.isnull().mean() * 100).round(2)
        # Create a DataFrame to display missing value counts and rates
        missing_value_table = pd.DataFrame({
            'Missing Value Count': missing_value_counts,
            'Missing Value Rate': missing_value_rates
        })

        # Print the missing value table
        print("Missing values :")
        print(missing_value_table, '\n')

    return

def have_uniformed(df):
    uniformed_columns = []
    for i in df.columns:
        unique_values = df[i].dropna().unique()
        
        if len(unique_values) == 1:
            uniformed_columns.append(i)
            
    if uniformed_columns:
        for i in uniformed_columns:
            print("The DataFrame contains columns with uniformed values:\n", i)  
        return True
    else:
        print("The DataFrame does not contain columns with uniformed values.\n")
        return False
    
def have_duplicate(df):
    duplicated_data = df.duplicated()
    num_duplicated_data = df.duplicated().sum()
    if duplicated_data.any():
        print(f"Number of duplicated records: {num_duplicated_data}\n")
        return True
    else:
        print("---> The DataFrame does not contain any dupplicated records.\n")
        return False