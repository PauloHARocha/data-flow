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
    experiment = 'acs2.0'
    columns = list(range(1,22))

    pre_p = PreProcess(experiment)
    # pre_p.initialize(dataset, normalization=False, columns=columns)

    #############################

    ###### Execute Algorithm ######
    algorithms = [
        KMeans(),
        FCMeans(fuzzy_c=2),
    ]
    k_min = 2
    k_max = 10
    n_sim = 30
    for algorithm in algorithms:
        exec_alg = ExecAlgorithm(pre_p.experiment, algorithm, k_max=k_max,n_sim=n_sim)
        exec_alg.run()
    # #############################


    # ####### Execute Metric ########
    # metrics = ['inter-cluster', 'cluster-separation', 'abgss',
    #            'edge-index', 'cluster-connectedness', 'intra-cluster',
    #            'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
    #            'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
    #            'silhouette', 'min-max-cut', 'gap']
    metrics = ['inter-cluster', 'intra-cluster', 'silhouette', 'ch-index']
    exe_met = ExecMetric(pre_p.experiment, metrics)
    
    # for algorithm in algorithms:
    #     exe_met.run(algorithm, k_min=k_min, k_max=k_max, n_sim=n_sim)
    ############################

    # linkages = ['single', 'complete', 'average', 'ward']
    # for link in linkages:
    #     exec_alg = ExecAlgorithm(pre_p.experiment, None, k_max=k_max)
    #     exec_alg.run_from_sklearn(linkage=link)
    
    # for link in linkages:
    #     exe_met.run_from_sklearn(link, k_min=k_min, k_max=k_max)



