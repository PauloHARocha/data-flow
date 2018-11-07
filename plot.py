import os, math
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np


# def plot_k_range_bp_all(experiment, algorithms, k_min, k_max, met):
#     k_range = range(k_min, k_max+1)
#     bp_alg_data = []
#     for algorithm in algorithms:
#         bp_data = []
#         for k in k_range:
#             met_path = f"booking/{experiment}/{algorithm.__str__()}/{k}/metrics"
#             met_file = f"{met_path}/{met}/results.csv"
#             met_data = pd.read_csv(met_file)
#             bp_data.append(met_data.values[:, 1])
#         bp_alg_data.append(bp_alg_data)

#     plot_path = f"booking/{experiment}/plots"
#     if not os.path.exists(plot_path):
#         os.mkdir(plot_path)
    
#     plt.figure()
#     figure_name = f"{plot_path}/{algorithm.__str__()}_{met}_k{k_min}-{k_max}_range_bp.png"
#     plt.title(f"{algorithm.__str__()} - {met}", fontsize=20)
    
#     axis = [-0.17, 0, 0.17]
#     for i in range(len(algorithms)):
#         print(i)
#         plt.boxplot(bp_alg_data[i], 0, positions=[x+axis[i] for x in k_range])
    
#     plt.tight_layout()
#     plt.savefig(figure_name)

def plot_k_range_all(experiment, algorithms, k_min, k_max, met):
    k_range = range(k_min, k_max+1)
    plot_mean = []
    plot_std = []
    df = pd.DataFrame()
    row_alg = []
    for algorithm in algorithms:
        print(algorithm)
        mean = []
        std = []
        for k in k_range:
            row_alg.append(f"{algorithm.__str__()}_k_{k}")
            met_path = f"booking/{experiment}/{algorithm.__str__()}/{k}/metrics"
            met_file = f"{met_path}/{met}/output.csv"
            if os.path.exists(met_file):
                met_data = pd.read_csv(met_file)
                mean.append(met_data.values[:,3][0])
                std.append(met_data.values[:, 4][0])
                # print(f"{algorithm.__str__()} / {k} / {met} / mean: {met_data.values[:,3][0]} / std: {met_data.values[:, 4][0]}")
            else:
                mean.append(math.inf)
                std.append(math.inf)
            
        plot_mean.append(mean)
        plot_std.append(std)
    df['algorithms'] = row_alg
    df.to_csv(f"booking/{experiment}/table.csv", index=False)
    plot_path = f"booking/{experiment}/plots"
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    
    plt.figure()
    figure_name = f"{plot_path}/{met}_k{k_min}-{k_max}_range.png"
    for p in range(len(plot_mean)): 
        plt.errorbar(k_range, plot_mean[p], yerr=plot_std[p], marker='o',
                          capthick=2, barsabove=True, label=algorithms[p].__str__())
    plt.title(f"{met}", fontsize=20)
    plt.xticks(k_range)
    plt.xlabel('k', fontsize=18)
    plt.ylabel('value', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name)


def plot_clusters(experiment, algorithm, k, n):
    colors = ['red', 'green', 'blue', 'yellow']

    plot_path = f"booking/{experiment}/plots"
    data_path = f"booking/{experiment}/data.csv"
    clusters_path = f"booking/{experiment}/{algorithm.__str__()}/{k}/clusters.csv"

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
                    plt.scatter(data[i][xf], data[i][yf], color=colors[class_[i]], s=10)
                plt.xlabel(features[xf])
                plt.ylabel(features[yf])
                plt.tight_layout()
                plt.savefig(
                    f"{plot_path}/clusters_k{k}_sim{n}_x{features[xf]}_y{features[yf]}.png")


def plot_data_distribution(experiment):
    data_path = f"booking/{experiment}/data.csv"

    data = pd.read_csv(data_path)
    features = data.columns.values[1:]

    figure_name = f"booking/{experiment}/plots/distribution_all_features.png"
    plt.figure()
    for i in range(len(features)):
        h = data.values[:, i + 1]
        h.sort()
        hmean = np.mean(h)
        hstd = np.std(h)
        pdf = stats.norm.pdf(h, hmean, hstd)
<<<<<<< HEAD
        plt.plot(h, pdf, label=features[i])
    plt.title(f"Data distribution", fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name)
||||||| merged common ancestors
        plt.figure()
        plt.plot(h, pdf)
        plt.title(f"{features[i]}", fontsize=20)
        plt.tight_layout()
        plt.savefig(figure_name)
=======
        plt.figure()
        plt.plot(h, pdf)
        plt.title(f"{features[i]}", fontsize=20)
        plt.tight_layout()
        plt.savefig(figure_name)


def cvs_range_all(experiment, algorithms, k_min, k_max, metrics):
    k_range = range(k_min, k_max+1)
    
    df = pd.DataFrame()
    
    for met in metrics:
        mean = []
        std = []
        row_alg = []
        for algorithm in algorithms:
            print(algorithm)
            
            for k in k_range:
                row_alg.append(f"{algorithm.__str__()}_k_{k}")
                met_path = f"booking/{experiment}/{algorithm.__str__()}/{k}/metrics"
                met_file = f"{met_path}/{met}/output.csv"
                if os.path.exists(met_file):
                    met_data = pd.read_csv(met_file)
                    mean.append(met_data.values[:,3][0])
                    std.append(met_data.values[:, 4][0])
                    print(f"{algorithm.__str__()} / {k} / {met} / mean: {met_data.values[:,3][0]} / std: {met_data.values[:, 4][0]}")
                else:
                    mean.append(math.inf)
                    std.append(math.inf)
        df['algorithms'] = row_alg
        df[met] = mean        
    
    
    
    
    df.to_csv(f"booking/{experiment}/table_{k_min}_{k_max}.csv", index=False)
    
    
def cvs_getBest(experiment, algorithms,k_min, k_max, metrics, evaluation):
    df = pd.read_csv(f"booking/{experiment}/table_{k_min}_{k_max}.csv")
    for m in range(len(metrics)):
        vals = df[metrics[m]].values
        if evaluation[m] == 'min':
            best = np.min(vals[vals != -np.inf])
            idx = np.where(vals == best)
        else:
            best = np.max(vals[vals != np.inf])
            idx = np.where(vals == best)

        idx = int(idx[0])
        vals = vals.tolist()
        vals[idx] = f'b {vals[idx]}'
        df[metrics[m]] = vals
    df.to_csv(f"booking/{experiment}/tableBest_{k_min}_{k_max}.csv", index=False)
    print(f"best: {vals[idx]} / idx: {idx}")
>>>>>>> 5590bde4812057cdce5c0175b935d0c4e55d90f4
