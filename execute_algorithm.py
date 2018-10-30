import os
import pandas as pd
from tqdm import tqdm


def exec_algorithm(experiment, algorithm, k_min, k_max, n_sim=30):
    data_file = pd.read_csv(f"booking/{experiment}/data.csv")

    algorithm_path = f"booking/{experiment}/{algorithm.__str__()}"
    if not os.path.exists(algorithm_path):
        os.mkdir(algorithm_path)

    k_range = range(k_min, k_max+1)
    for k in tqdm(k_range, desc="k: "):
        for n in range(n_sim):
            data = data_file.values[:,1:] #select values from data
            algorithm.fit(data=data, k=k)  # run algorithm

            k_path = f"{algorithm_path}/{k}"
            if not os.path.exists(k_path):
                os.mkdir(k_path)

            centroids_file = f"{k_path}/centroids_sim_{n}.csv" #save centroids
            save_centroids = pd.DataFrame(algorithm.centroids)
            save_centroids = save_centroids.transpose()
            save_centroids.to_csv(centroids_file)

    config = pd.DataFrame()
    config['k_min'] = [k_min]
    config['k_max'] = [k_max]
    config['n_sim'] = [n_sim]
    if hasattr(algorithm, 'tol'): config['tol'] = [algorithm.tol]
    if hasattr(algorithm, 'n_iter'): config['n_iter'] = [algorithm.n_iter]
    if hasattr(algorithm, 'swarm_size'): config['swarm_size'] = [algorithm.swarm_size]
    if hasattr(algorithm, 'trials_limit'): config['trials_limit'] = [algorithm.trials_limit]

    config.to_csv(f"{algorithm_path}/config.csv")  # Save data configuration


    
