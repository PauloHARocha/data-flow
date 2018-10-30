import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_k_range(experiment, algorithm, k_min, k_max, met):
    k_range = range(k_min, k_max+1)
    mean = []
    std = []
    for k in k_range:
        met_path = f"booking/{experiment}/{algorithm.__str__()}/{k}/metrics"
        met_file = f"{met_path}/{met}/output.csv"
        met_data = pd.read_csv(met_file)
        mean.append(met_data.values[:,3][0])
        std.append(met_data.values[:, 4][0])

    plot_path = f"booking/{experiment}/plots"
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    
    plt.figure()
    figure_name = f"{plot_path}/{algorithm.__str__()}_{met}_k{k_min}-{k_max}_range.png"
    plt.title(f"{algorithm.__str__()} - {met}", fontsize=20)
    plt.errorbar(k_range, mean, yerr=std, marker='o',
                         ecolor='b', capthick=2, barsabove=True)
    plt.xticks(k_range)
    plt.xlabel('k', fontsize=18)
    plt.ylabel('value', fontsize=18)
    plt.tight_layout()
    plt.savefig(figure_name)
