import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from algorithms.clustering.metrics import Metrics


def exec_metric(experiment, algorithm, k_min, k_max, n_sim, metric):

    alg_name = algorithm.__str__()
    data_file = pd.read_csv(f"booking/{experiment}/data.csv")
    
    
    k_range = range(k_min, k_max+1)
    for k in tqdm(k_range, desc=f"{metric}:"):
        met_path = f"booking/{experiment}/{alg_name}/{k}/metrics"
        if not os.path.exists(met_path):
            os.mkdir(met_path)
        met_results = []
        for n in range(n_sim):
            centroid_file = pd.read_csv(
                f"booking/{experiment}/{alg_name}/{k}/centroids_sim_{n}.csv")
            
            met_results.append(Metrics.evaluate(
                metric, data_file.values[:, 1:], centroid_file.values[:, 1:]
                , algorithm=algorithm))

        met_path = f"{met_path}/{metric}"
        if not os.path.exists(met_path):
            os.mkdir(met_path)
        results_file = f"{met_path}/results.csv"
        save_met_results = pd.DataFrame(met_results)
        save_met_results.to_csv(results_file)

        
        output_file = f"{met_path}/output.csv"
        save_met_output = pd.DataFrame()
        save_met_output['algorithm'] = [alg_name]
        save_met_output['k'] = [k]
        save_met_output['mean'] = [np.mean(met_results)]
        save_met_output['std'] = [np.std(met_results)]
        save_met_output.to_csv(output_file)

