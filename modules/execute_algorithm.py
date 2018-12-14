import os
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

class ExecAlgorithm():

    def __init__(self, experiment, algorithm, k_min=2, k_max=2, n_sim=30):
        self.experiment = experiment
        self.algorithm = algorithm
        self.k_min = k_min
        self.k_min = k_min
        self.k_max = k_max
        self.n_sim = n_sim

    @property
    def exp_path(self):
        return f"booking/{self.experiment}"

    def run(self):
        data_file = pd.read_csv(f"{self.exp_path}/data.csv")
        data = data_file.values  # select values from data

        algorithm_path = f"{self.exp_path}/{self.algorithm.__str__()}"
        self.check_path(algorithm_path)
        k_range = range(self.k_min, self.k_max+1)

        for k in tqdm(k_range, desc="k: "):
            k_path = f"{algorithm_path}/{k}"
            self.check_path(k_path)
                    
            clusters_df = pd.DataFrame()
            
            for n in range(self.n_sim):
                self.algorithm.fit(data=data, k=k)  # run algorithm
                # self.save_centroids(k_path, n, algorithm.centroids)
                clusters_df = self.save_clusters(clusters_df, n, self.algorithm, data)

            clusters_path = f"{k_path}/{self.algorithm.__str__()}_k{k}_clusters_labels.csv"
            clusters_df.to_csv(clusters_path, index=False)

        self.save_config(self.k_min, self.k_max, self.n_sim, self.algorithm, algorithm_path)

            
    def run_from_sklearn(self, linkage):
        data_file = pd.read_csv(f"{self.exp_path}/data.csv")
        data = data_file.values  # select values from data
        alg_name = f'AGL-{linkage}'
        algorithm_path = f"{self.exp_path}/{alg_name}"
        self.check_path(algorithm_path)
        k_range = range(self.k_min, self.k_max+1)

        for k in tqdm(k_range, desc="k: "):
            k_path = f"{algorithm_path}/{k}"
            self.check_path(k_path)
                    
            algorithm = AgglomerativeClustering(linkage=linkage, n_clusters=k)
            algorithm.fit(data)    
            
            clusters_df = pd.DataFrame()
            clusters_df['sim_0'] = algorithm.labels_
            clusters_path = f"{k_path}/{alg_name}_k{k}_clusters_labels.csv"
            clusters_df.to_csv(clusters_path, index=False)

        self.save_config(self.k_min, self.k_max, 1, linkage, algorithm_path)
    # def save_centroids(self, k_path, n, centroids):
    #     centroids_file = f"{k_path}/centroids_sim_{n}.csv"  # save centroids

    #     centroids_df = pd.DataFrame(centroids)
    #     centroids_df = centroids_df.transpose()
    #     centroids_df.to_csv(centroids_file, index=False)
    
    def save_clusters(self, clusters_df, n, algorithm, data):
        class_ = []
        for dataPoint in data:
            class_.append(algorithm.predict(dataPoint))
        clusters_df[f"sim_{n}"] = class_
        return clusters_df

    def save_config(self, k_min, k_max, n_sim, algorithm, algorithm_path):
        config = pd.DataFrame()
        config['k_min'] = [k_min]
        config['k_max'] = [k_max]
        config['n_sim'] = [n_sim]
        if hasattr(algorithm, 'tol'):
            config['tol'] = [algorithm.tol]
        if hasattr(algorithm, 'n_iter'):
            config['n_iter'] = [algorithm.n_iter]
        if hasattr(algorithm, 'swarm_size'):
            config['swarm_size'] = [algorithm.swarm_size]
        if hasattr(algorithm, 'trials_limit'):
            config['trials_limit'] = [algorithm.trials_limit]

        config.to_csv(f"{algorithm_path}/config.csv")  # Save data configuration

    def check_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
