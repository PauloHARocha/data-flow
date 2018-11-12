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
        features = data.columns.values[:]
        res = []
        for i in range(len(features)):
            h = data.values[:, i]
            h.sort()
            hmean = np.mean(h)
            hstd = np.std(h)
            pdf = stats.norm.pdf(h, hmean, hstd)
            res.append({
                'legend': features[i],
                'labelX': 'h',
                'labelY': 'pdf',
                'x': {
                    'values': h.tolist()
                },
                'y': {
                    'values': pdf.tolist()
                }
            })
        return res


    def plot_clusters(self, algorithm, k=2, n=0):

        data_path = f"{self.exp_path}/data.csv"
        clusters_path = f"{self.exp_path}/{algorithm}/{k}/clusters.csv"

        data = pd.read_csv(data_path)
        features = data.columns.values[:]
        data = data.values[:, :]
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
                                'values': [float(d[xf]) for d in data[id_[i]]]
                            },
                            'y': {
                                'values': [float(d[yf]) for d in data[id_[i]]]
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
                    'labelX': 'k',
                    'labelY': 'value',
                    'x': {
                        'values': list(k_range)
                    },
                    'y': {
                        'values': mean
                    }
                })
            res.append({
                'legend': met,
                'algorithms': alg
            })

        return res

    def gen_corr_df(self):
        k_range = range(self.k_min, self.k_max+1)
        cd = pd.DataFrame()
        for met in self.metrics:
            for algorithm in self.algorithms:
                mean = []
                for k in k_range:
                    met_path = f"{self.exp_path}/{algorithm}/{k}/metrics"
                    met_file = f"{met_path}/{met}/output.csv"
                    if os.path.exists(met_file):
                        met_data = pd.read_csv(met_file)
                        mean.append(met_data.values[:, 3][0])
            cd[met] = mean
            
        cd.to_csv(f"{self.exp_path}/before_corr_metrics.csv", index=False)
        cd = cd.corr()
        cd.to_csv(f"{self.exp_path}/corr_metrics.csv", index=False)

        
        data = cd.values
        
        column_labels = cd.columns
        row_labels = cd.index

        fig, ax = plt.subplots()

        im = ax.imshow(data, cmap=plt.cm.coolwarm)
        fig.colorbar(im)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(column_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        
        # ... and label them with the respective list entries
        ax.set_xticklabels(column_labels)
        ax.set_yticklabels(row_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(row_labels)):
            for j in range(len(column_labels)):
                ax.text(j, i, round(data[i, j].astype(np.float32),2),
                            ha="center", va="center", color="w", fontsize=5)

        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        fig_name = f"{self.exp_path}/corr_matrix_metrics.png"
        plt.savefig(fig_name, dpi=(200))

        return fig_name
