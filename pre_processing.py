import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

def pre_process(dataset, experiment, rows=[], columns=[], normalization=False):
    ds = { #Available datasets
        'iris': pd.DataFrame(datasets.load_iris().data[:, :]),
        'wine': pd.DataFrame(datasets.load_wine().data[:, :]),
        'alzheimer': pd.read_csv("datasets/dataPhDAlzheimerSemNomes.csv")}

    if rows and columns: #Select rows and columns
        data = ds[dataset].iloc[rows, columns]
    elif rows:
        data = ds[dataset].iloc[rows, :]
    elif columns:
        data = ds[dataset].iloc[:, columns]
    else:
        data = ds[dataset].iloc[:, :]
    
    if normalization: #Normalize dataset
        std = MinMaxScaler()
        data = std.fit_transform(data.values)
        data = pd.DataFrame(data)
    
    data_path = f"booking/{experiment}"
    if not os.path.exists(data_path): #Create directory of experiment
        os.mkdir(data_path)

    data.to_csv(f"{data_path}/data.csv")#Save data

    config = pd.DataFrame()
    config['dataset'] = [dataset]
    config['rows'] = [rows]
    config['columns'] = [columns]
    config['normalization'] = [normalization]
    
    config.to_csv(f"{data_path}/config.csv")#Save data configuration
    

