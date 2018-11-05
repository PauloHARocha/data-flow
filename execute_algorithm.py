import os
import pandas as pd
from tqdm import tqdm


def exec_algorithm(experiment, algorithm, k_min, k_max, n_sim=30):
    data_file = pd.read_csv(f"booking/{experiment}/data.csv")

    algorithm_path = f"booking/{experiment}/{algorithm.__str__()}"
    check_path(algorithm_path)

    k_range = range(k_min, k_max+1)
    for k in tqdm(k_range, desc="k: "):
        k_path = f"{algorithm_path}/{k}"
        check_path(k_path)
        data = data_file.values[:,1:] #select values from data
        
        clusters_df = pd.DataFrame()
        clusters_df['data_points'] = data_file.values[:, 0]
        clusters_path = f"{k_path}/clusters.csv"
        clusters_df.to_csv(clusters_path, index=False)

        for n in range(n_sim):
            algorithm.fit(data=data, k=k)  # run algorithm

            save_centroids(k_path, n, algorithm.centroids)
            save_clusters(clusters_path, n, algorithm, data)

    save_config(k_min, k_max, n_sim, algorithm, algorithm_path)

def save_centroids(k_path, n, centroids):
    centroids_file = f"{k_path}/centroids_sim_{n}.csv"  # save centroids

    centroids_df = pd.DataFrame(centroids)
    centroids_df = centroids_df.transpose()
    centroids_df.to_csv(centroids_file)

def save_clusters(clusters_path, n, algorithm, data):
    clusters_df = pd.read_csv(clusters_path)
    class_ = []
    for dataPoint in data:
        class_.append(algorithm.predict(dataPoint))
    clusters_df[f"sim_{n}"] = class_
    clusters_df.to_csv(clusters_path, index=False)

def save_config(k_min, k_max, n_sim, algorithm, algorithm_path):
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


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
