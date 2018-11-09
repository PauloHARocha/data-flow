import os
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np


class Plot():

    def __init__(self, experiment='', algorithms=[], metrics=[], k_min=2, k_max=2):
        self.experiment = experiment
        self.algorithms = algorithms
        self.metrics = metrics
        self.k_min = k_min
        self.k_max = k_max
    
    @property
    def exp_path(self):
        return f"booking/{self.experiment}"

    def plot_data_distribution(self):
        data = pd.read_csv(f"{self.exp_path}/data.csv")
        features = data.columns.values[1:]
        res = []
        for i in range(len(features)):
            h = data.values[:, i + 1]
            h.sort()
            hmean = np.mean(h)
            hstd = np.std(h)
            pdf = stats.norm.pdf(h, hmean, hstd)
            res.append({
                'feature': features[i],
                'x': {
                    'label': 'h',
                    'values': h.tolist()
                },
                'y': {
                    'label': 'pdf',
                    'values': pdf.tolist()
                }
            })
        return res
