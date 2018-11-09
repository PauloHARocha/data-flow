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


    def plot_clusters(self, algorithm, k=2, n=0):

        data_path = f"{self.exp_path}/data.csv"
        clusters_path = f"{self.exp_path}/{algorithm}/{k}/clusters.csv"

        data = pd.read_csv(data_path)
        features = data.columns.values[1:]
        data = data.values[:, 1:]

        class_ = pd.read_csv(clusters_path)
        class_ = class_.values[:, n+1]
        id_ = []
        for i in range(k):
            id_.append(np.where(class_ == i)[0].tolist())
        res = []
        for xf in range(len(features)):
            for yf in range(len(features)):
                if xf < yf:
                    clusters = []
                    for i in range(len(id_)):
                        clusters.append({
                            'labelX': features[xf],
                            'labelY': features[yf],
                            'legend': len(id_[i]),
                            'x': {
                                'values': [int(d[xf]) for d in data[id_[i]]]
                            },
                            'y': {
                                'values': [int(d[yf]) for d in data[id_[i]]]
                            }
                        })
                    res.append(clusters)
        return res
    
    def plot_k_range(self):
        k_range = range(self.k_min, self.k_max+1)
        res = []
        for met in self.metrics:
            alg = []
            for algorithm in self.algorithms:
                mean = []
                std = []
                for k in k_range:
                    met_path = f"{self.exp_path}/{algorithm}/{k}/metrics"
                    met_file = f"{met_path}/{met}/output.csv"
                    if os.path.exists(met_file):
                        met_data = pd.read_csv(met_file)
                        mean.append(met_data.values[:, 3][0])
                        std.append(met_data.values[:, 4][0])
                    else:  # In case have missing values
                        mean.append(math.inf)
                        std.append(math.inf)
                alg.append({
                    'legend': algorithm,
                    'x': {
                        'values': list(k_range)
                    },
                    'y': {
                        'values': mean
                    }
                })
            res.append({
                'labelX': 'k',
                'labelY': 'value',
                'legend': met,
                'algorithms': alg
            })

        return res
