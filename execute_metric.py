import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from algorithms.clustering.metrics import Metrics

class ExecMetric():

    def __init__(self, experiment, metrics):
        self.experiment = experiment
        self.metrics = metrics

    @property
    def exp_path(self):
        return f"booking/{self.experiment}"

    def run(self, algorithm, k_min=2, k_max=2, n_sim=30):

        alg_name = algorithm.__str__()
        data_file = pd.read_csv(f"{self.exp_path}/data.csv")
        
        k_range = range(k_min, k_max+1)
        for met in self.metrics:
            for k in tqdm(k_range, desc=f"{met}:"):
                metrics_path = f"{self.exp_path}/{alg_name}/{k}/metrics"
                print(metrics_path)
                self.check_path(metrics_path)

                met_results = []
                for n in range(n_sim):
                    centroid_file = pd.read_csv(
                        f"{self.exp_path}/{alg_name}/{k}/centroids_sim_{n}.csv")
                    
                    met_results.append(Metrics.evaluate(
                        met, data_file.values[:, 1:], centroid_file.values[:, 1:]
                        , algorithm=algorithm))

                met_path = f"{metrics_path}/{met}"
                self.check_path(met_path)

                self.save_results(met_path, met_results)
                self.save_output(met_path, alg_name, k, met_results)

    def save_results(self, path, results):
        filename = f"{path}/results.csv"
        results = pd.DataFrame(results)
        results.to_csv(filename)

    def save_output(self, path, alg_name, k, results):
        filename = f"{path}/output.csv"
        output = pd.DataFrame()
        output['algorithm'] = [alg_name]
        output['k'] = [k]
        output['mean'] = [np.mean(results)]
        output['std'] = [np.std(results)]
        output.to_csv(filename)

    def check_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
