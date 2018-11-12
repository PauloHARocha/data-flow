import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

class PreProcess():

    def __init__(self, experiment):
        self.experiment = experiment
        self.ds = {  # Available datasets
            'iris': pd.DataFrame(datasets.load_iris().data[:, :]),
            'wine': pd.DataFrame(datasets.load_wine().data[:, :]),
            'acs': pd.read_csv("datasets/acs_16_5_county.csv")
            # 'alzheimer': pd.read_csv("datasets/dataPhDAlzheimerSemNomes.csv")
        }
    
    @property
    def exp_path(self):
        return f"booking/{self.experiment}"

    def initialize(self, ds_name, rows=[], columns=[], normalization=False):
        
        self.data = self.selectData(ds_name, rows, columns)
        
        if normalization: #Normalize dataset
            std = MinMaxScaler()
            self.data = std.fit_transform(self.data.values)
            self.data = pd.DataFrame(self.data)
        
        if not os.path.exists(self.exp_path): #Create directory of experiment
            os.mkdir(self.exp_path)

        self.data.to_csv(f"{self.exp_path}/data.csv")#Save data

        config = pd.DataFrame()
        config['dataset'] = [ds_name]
        config['rows'] = [rows]
        config['columns'] = [columns]
        config['normalization'] = [normalization]
        
        config.to_csv(f"{self.exp_path}/config.csv")#Save data configuration
        

    def selectData(self, ds_name, rows, columns):
        if rows and columns:  # Select rows and columns
            data = self.ds[ds_name].iloc[rows, columns]
        elif rows:
            data = self.ds[ds_name].iloc[rows, :]
        elif columns:
            data = self.ds[ds_name].iloc[:, columns]
        else:
            data = self.ds[ds_name].iloc[:, :]

        return data
