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
        self.check_path(f"{self.exp_path}/plots")

    @property
    def exp_path(self):
        return f"booking/{self.experiment}"
    
    def check_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
    
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
        for met in self.metrics:
            plot_mean = []
            plot_std = []
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
        id_ = []
        for i in range(k):
            id_.append(np.where(class_ == i)[0].tolist())
            
        for xf in range(len(features)):
            for yf in range(len(features)):
                if xf < yf:
                    plt.figure()
                    for i in range(len(id_)):
                        plt.scatter([d[xf] for d in data[id_[i]]],
                                    [d[yf] for d in data[id_[i]]],
                                    color=self.colors[i],
                                    label=f"{len(id_[i])}" , 
                                    s=10)
                    plt.xlabel(features[xf])
                    plt.ylabel(features[yf])
                    plt.legend()
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
                        mean.append(
                            f"{format(met_data.values[:,3][0], '.4f')} ({format(met_data.values[:, 4][0], '.4f')})")
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

    def gen_corr_df(self):
        k_range = range(self.k_min, self.k_max+1)
        cd = pd.DataFrame()
        for met in self.metrics:
            mean = []
            for algorithm in self.algorithms:
                for k in k_range:
                    met_path = f"{self.exp_path}/{algorithm}/{k}/metrics"
                    met_file = f"{met_path}/{met}/output.csv"
                    if os.path.exists(met_file):
                        met_data = pd.read_csv(met_file)
                        mean.append(met_data.values[:, 3][0])
            cd[met] = mean

        # cd.to_csv(f"{self.exp_path}/before_corr_metrics.csv", index=False)
        cd = cd.corr()
        # cd.to_csv(f"{self.exp_path}/corr_metrics.csv", index=False)

        data = cd.values

        column_labels = cd.columns
        row_labels = cd.index

        return self.get_corr_fig(data, column_labels, row_labels, 'corr_matrix_metrics', 'Correlation Matrix')

    def plot_class_corr(self, algorithm, k=2, n=0):

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

        for i in range(len(id_)):
            _class_values = [d for d in data[id_[i]]]
            df = pd.DataFrame(_class_values)
            df.columns = features
            df = df.corr()
            dt = df.values

            column_labels = df.columns
            row_labels = df.index
            res.append(self.get_corr_fig(dt, column_labels,
                                         row_labels, f"corr_matrix_c{i}", f"Cluster {i}"))
        return res

    def get_corr_fig(self, data, column_labels, row_labels, name, title):
        fig, ax = plt.subplots()

        im = ax.imshow(data, cmap='coolwarm')
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
                ax.text(j, i, round(data[i, j].astype(np.float32), 2),
                        ha="center", va="center", color="black", fontsize=5)

        ax.set_title(title)
        plt.tight_layout()
        fig_name = f"{self.exp_path}/{name}.png"
        plt.savefig(fig_name, dpi=(200))

        return fig_name

    def plot_intra_inter(self):
        k_range = range(4, 5)
        plot_final = []
        for met in ['inter-cluster', 'intra-cluster']:
            
            mean = []
            
            for algorithm in self.algorithms:
                for k in k_range:
                    met_path = f"{self.exp_path}/{algorithm.__str__()}/{k}/metrics"
                    met_file = f"{met_path}/{met}/output.csv"
                    if os.path.exists(met_file):
                        met_data = pd.read_csv(met_file)
                        mean.append(met_data.values[:,3][0])
                                    
                
            plot_final.append(mean)
        
        plot_path = f"{self.exp_path}/plots"
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        
        figure_name = f"{plot_path}/intra-inter_k4.png"
        plt.figure()
        for a in range(len(self.algorithms)):
            plt.plot(plot_final[0][a], plot_final[1][a],
                     marker='o', label=self.algorithms[a].__str__())
        # for p in range(len(plot_mean)): 
        #     plt.errorbar(k_range, plot_mean[p], yerr=plot_std[p], marker='o',
        #                     capthick=2, barsabove=True, 
        #                     label=self.algorithms[p].__str__())
        # plt.title(f"{met}", fontsize=20)
        # plt.xticks(k_range)
        plt.xlabel('inter-cluster', fontsize=18)
        plt.ylabel('intra-cluster', fontsize=18)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_name)
