import os
import shutil
import sys 
import pandas as pd 
import random

LEN_DATA = 40000
def fill_panda(data):
    colx = data.columns
    while len(data) < LEN_DATA:
        row = pd.Series([None] * len(data.columns), index=data.columns)
        data = data.append(row, ignore_index=True)
    if len(data) > LEN_DATA:
        start = random.randint(0,len(data) - LEN_DATA)
        data = data.iloc[start: start+LEN_DATA]
    return data

def merge_csv_files(folder_name):
    # Create the output folder if it doesn't exist
    output_folder = os.path.join('./', "GLOBAL")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Loop through all subdirectories and find CSV files
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.endswith(".csv"):
                # Copy the CSV file to the output folder
                source_file = os.path.join(root, file)

                panditas = pd.read_csv(source_file, index_col=0)
                print(panditas.head())
                panditas = fill_panda(panditas)
                #destination_file = os.path.join(output_folder, file)
                #panditas.to_csv(destination_file)
                del panditas
        sys.exit(0)

    print("CSV files merged into GLOBAL2 folder.")

# Example usage:
folder_name = sys.argv[1]
merge_csv_files(folder_name)
