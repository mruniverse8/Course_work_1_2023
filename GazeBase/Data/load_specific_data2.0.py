import csv
import sys
import os
import pandas as pd
#WE ARE DROPPING INFO
output_dir = "./SPECIFIC_DATA_TEX"
def divide_csv(csv_path, x):
    #print(csv_path)
    csv_path2 = os.path.join(origin, csv_path)
    data = pd.read_csv(csv_path2)
    data = data.drop(columns='lab')
    data = data.drop(columns='xT')
    data = data.drop(columns='yT')
    data = data.drop(columns='dP')
    num_rows = len(data)
    chunk_size = x * 1000
    chunks = [data[i:i+chunk_size] for i in range(0, num_rows, chunk_size)]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_dir, f"{csv_path}_{i}.csv")
        #print(output_path)
        chunk.to_csv(output_path, index=False)
    #print("just all!")

fd = open("table_TEX.txt", "r")
origin = './GLOBAL'
num_rows = 6
for line in fd:
    divide_csv(line[:-1], num_rows)
print("oks")
