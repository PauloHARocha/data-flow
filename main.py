from pre_processing import pre_process
from execute_algorithm import exec_algorithm
from execute_metric import exec_metric
from plot import plot_k_range, plot_k_range_all
from algorithms.clustering.ABCC import ABCC
from algorithms.clustering.kmeans import KMeans
from algorithms.clustering.fcmeans import FCMeans
from algorithms.clustering.metrics import Metrics


if __name__ == '__main__':
    ####### Pre-processing ########
    dataset = 'iris'
    experiment = 'iris-abcc'
    rows = []
    columns = []
    normalization = True

    # pre_process(dataset, experiment, rows, columns, normalization) 
    #############################

    ###### Execute Algorithm ######
    algorithms = [
        ABCC(n_iter=100, swarm_size=50, trials_limit=10, metric_type='max', evaluation_metric='ch-index'),
        ABCC(n_iter=100, swarm_size=50, trials_limit=10, metric_type='min', evaluation_metric='davies-bouldin'),
        ABCC(n_iter=100, swarm_size=50, trials_limit=10, metric_type='min', evaluation_metric='intra-cluster'),
        FCMeans(),
    ]
    # algorithm = KMeans()
    k_min = 2
    k_max = 8
    
    n_sim = 5
    # for alg in algorithms:
    #     exec_algorithm(experiment, alg, k_min, k_max, n_sim)
    # #############################


    # ####### Execute Metric ########
    metrics = ['inter-cluster', 'cluster-separation', 'abgss',
               'edge-index', 'cluster-connectedness', 'intra-cluster',
               'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
               'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
               'silhouette', 'min-max-cut']

    # for met in metrics:
    #     for algorithm in algorithms:
    #         exec_metric(experiment, algorithm, k_min, k_max, n_sim, met)
    ############################



    ########### Plot ############


    for met in metrics:
        plot_k_range_all(experiment, algorithms, k_min, k_max, met)
    ############################

