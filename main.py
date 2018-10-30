from pre_processing import pre_process
from execute_algorithm import exec_algorithm
from execute_metric import exec_metric
from plot import plot_k_range
from algorithms.clustering.ABCC import ABCC
from algorithms.clustering.kmeans import KMeans
from algorithms.clustering.metrics import Metrics


if __name__ == '__main__':
    ####### Pre-processing ########
    dataset = 'iris'
    experiment = 'iris1'
    rows = []
    columns = []
    normalization = True

    # pre_process(dataset, experiment, rows, columns, normalization) 
    #############################

    ###### Execute Algorithm ######
    algorithm = ABCC(n_iter=100, swarm_size=50, trials_limit=10, evaluation_metric=Metrics.min_max_cut)
    # algorithm = KMeans()
    k_min = 2
    k_max = 3
    
    n_sim = 5

    exec_algorithm(experiment, algorithm, k_min, k_max, n_sim)
    #############################

    ####### Execute Metric ########
    metrics = ['ch-index', 'silhouette']

    # for met in metrics:
    #     exec_metric(experiment, algorithm, k_min, k_max, n_sim, met)
    #############################

    ########### Plot ############
    for met in metrics:
        plot_k_range(experiment, algorithm, k_min, k_max, met)

    #############################

