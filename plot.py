import os, math
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
        self.colors = ['red', 'green', 'blue',
                                'yellow', 'brown', 'gray', 'orange', 'purple']
    
    @property
    def exp_path(self):
        return f"booking/{self.experiment}"
    
    def plot_data_distribution(self): 
        data = pd.read_csv(f"{self.exp_path}/data.csv")
        features = data.columns.values[1:]
        figure_name = f"{self.exp_path}/plots/distribution_all_features.png"
        plt.figure()
        for i in range(len(features)):
            h = data.values[:, i + 1]
            h.sort()
            hmean = np.mean(h)
            hstd = np.std(h)
            pdf = stats.norm.pdf(h, hmean, hstd)
            plt.plot(h, pdf, label=features[i])
        plt.title(f"Data distribution", fontsize=20)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_name)

    def plot_k_range(self):
        k_range = range(self.k_min, self.k_max+1)
        plot_mean = []
        plot_std = []
        for met in self.metrics:
            for algorithm in self.algorithms:
                mean = []
                std = []
                for k in k_range:
                    met_path = f"{self.exp_path}/{algorithm.__str__()}/{k}/metrics"
                    met_file = f"{met_path}/{met}/output.csv"
                    if os.path.exists(met_file):
                        met_data = pd.read_csv(met_file)
                        mean.append(met_data.values[:,3][0])
                        std.append(met_data.values[:, 4][0])
                    else: #In case have missing values
                        mean.append(math.inf)
                        std.append(math.inf)            
                plot_mean.append(mean)
                plot_std.append(std)
            
        plot_path = f"{self.exp_path}/plots"
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        
        figure_name = f"{plot_path}/{met}_k{self.k_min}-{self.k_max}_range.png"
        plt.figure()
        for p in range(len(plot_mean)): 
            plt.errorbar(k_range, plot_mean[p], yerr=plot_std[p], marker='o',
                            capthick=2, barsabove=True, 
                            label=self.algorithms[p].__str__())
        plt.title(f"{met}", fontsize=20)
        plt.xticks(k_range)
        plt.xlabel('k', fontsize=18)
        plt.ylabel('value', fontsize=18)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_name)


    def plot_clusters(self, algorithm, k=2, n=0):

        plot_path = f"{self.exp_path}/plots"
        data_path = f"{self.exp_path}/data.csv"
        clusters_path = f"{self.exp_path}/{algorithm.__str__()}/{k}/clusters.csv"

        data = pd.read_csv(data_path)
        features = data.columns.values[1:]
        data = data.values[:, 1:]

        class_ = pd.read_csv(clusters_path)
        class_ = class_.values[:, n+1]
        for xf in range(len(features)):
            for yf in range(len(features)):
                if xf < yf:
                    plt.figure()
                    for i in range(len(class_)):
                        plt.scatter(data[i][xf], data[i][yf], color=self.colors[class_[i]], s=10)
                    plt.xlabel(features[xf])
                    plt.ylabel(features[yf])
                    plt.tight_layout()
                    plt.savefig(
                        f"{plot_path}/clusters_k{k}_sim{n}_x{features[xf]}_y{features[yf]}.png")


    def cvs_k_range(self):
        k_range = range(self.k_min, self.k_max+1)
        
        df = pd.DataFrame()
        for met in self.metrics:
            mean = []
            std = []
            row_alg = []
            for algorithm in self.algorithms:
                for k in k_range:
                    row_alg.append(f"{algorithm.__str__()}_k_{k}")
                    met_path = f"{self.exp_path}/{algorithm.__str__()}/{k}/metrics"
                    met_file = f"{met_path}/{met}/output.csv"
                    if os.path.exists(met_file):
                        met_data = pd.read_csv(met_file)
                        mean.append(met_data.values[:,3][0])
                        std.append(met_data.values[:, 4][0])
                        print(f"{algorithm.__str__()} / {k} / {met} / mean: {met_data.values[:,3][0]} / std: {met_data.values[:, 4][0]}")
                    else:  # In case have missing values
                        mean.append(math.inf)
                        std.append(math.inf)
            df['algorithms'] = row_alg
            df[met] = mean        
        
        df.to_csv(f"{self.exp_path}/table_{self.k_min}_{self.k_max}.csv", index=False)
    
    
    def cvs_k_range_best(self, evaluation):
        df = pd.read_csv(f"{self.exp_path}/table_{self.k_min}_{self.k_max}.csv")
        for m in range(len(self.metrics)):
            vals = df[self.metrics[m]].values
            if evaluation[m] == 'min':
                best = np.min(vals[vals != -np.inf])
                idx = np.where(vals == best)
            else:
                best = np.max(vals[vals != np.inf])
                idx = np.where(vals == best)

            idx = int(idx[0])
            vals = vals.tolist()
            vals[idx] = f'b {vals[idx]}'
            df[self.metrics[m]] = vals

        df.to_csv(f"{self.exp_path}/tableBest_{self.k_min}_{self.k_max}.csv", index=False)
        print(f"best: {vals[idx]} / idx: {idx}")
