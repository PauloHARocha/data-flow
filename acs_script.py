from modules.pre_processing import PreProcess
from modules.execute_algorithm import ExecAlgorithm
from modules.execute_metric import ExecMetric
from modules.plot import Plot
from algorithms.clustering.ABCC import ABCC
from algorithms.clustering.kmeans import KMeans
from algorithms.clustering.fcmeans import FCMeans
from algorithms.clustering.metrics import Metrics


if __name__ == '__main__':
    ####### Pre-processing ########
    dataset = 'acs'
    experiment = 'acs'
    columns = list(range(1,22))

    pre_p = PreProcess(experiment)
    pre_p.initialize(dataset, columns=columns)

    # plot = Plot(pre_p.experiment)

    # plot.plot_data_distribution()

    #############################

    ###### Execute Algorithm ######
    algorithms = [
        KMeans(),
        FCMeans()
    ]
    k_min = 2
    k_max = 10
    n_sim = 1
    exec_alg = ExecAlgorithm(pre_p.experiment, algorithms, k_max=k_max,n_sim=n_sim)
    # exec_alg.run()
    # #############################


    # ####### Execute Metric ########
    metrics = ['inter-cluster', 'cluster-separation', 'abgss',
               'edge-index', 'cluster-connectedness', 'intra-cluster',
               'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
               'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
               'silhouette', 'min-max-cut']#, 'gap']
    # metrics = ['gap']
    exe_met = ExecMetric(pre_p.experiment, metrics)
    
    for algorithm in exec_alg.algorithms:
        exe_met.run(algorithm, k_min=k_min, k_max=k_max, n_sim=n_sim)
    ############################

    plot = Plot(pre_p.experiment, exec_alg.algorithms, exe_met.metrics, k_min=k_min, k_max=k_max)

    plot.plot_data_distribution()
    plot.plot_k_range()

    ############################

