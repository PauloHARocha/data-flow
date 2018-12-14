from modules.pre_processing import PreProcess
from modules.execute_algorithm import ExecAlgorithm
from modules.execute_metric import ExecMetric
from modules.plot import Plot
from algorithms.clustering.ABCC import ABCC
from algorithms.clustering.PSOC import PSOC
from algorithms.clustering.kmeans import KMeans
from algorithms.clustering.fcmeans import FCMeans
from algorithms.clustering.metrics import Metrics


if __name__ == '__main__':
    ####### Pre-processing ########
    dataset = 'iris'
    experiment = 'iris-abcc'
    columns = []

    pre_p = PreProcess(experiment)
    # pre_p.initialize(dataset, normalization=True)

    # plot = Plot(pre_p.experiment)

    # plot.plot_data_distribution()

    #############################

    ###### Execute Algorithm ######
    algorithms = [
        # KMeans(),
        FCMeans(),
        # ABCC(n_iter=100, swarm_size=50, trials_limit=10, metric_type='max', evaluation_metric='ch-index'),
        # ABCC(n_iter=100, swarm_size=50, trials_limit=10, metric_type='min', evaluation_metric='davies-bouldin'),
        # ABCC(n_iter=100, swarm_size=50, trials_limit=10, metric_type='min', evaluation_metric='intra-cluster'),
        # PSOC(n_iter=100, swarm_size=50, evaluation_metric='intra-cluster')
    ]
    k_min = 2
    k_max = 8
    n_sim = 30
    exec_alg = ExecAlgorithm(
        pre_p.experiment, algorithms, k_max=k_max, n_sim=n_sim)
    # exec_alg.run()
    # #############################

    # ####### Execute Metric ########
    # metrics = ['inter-cluster', 'cluster-separation', 'abgss',
    #            'edge-index', 'cluster-connectedness', 'intra-cluster',
    #            'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
    #            'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
    #            'silhouette', 'min-max-cut', 'gap']
    metrics = ['inter-cluster', 'intra-cluster','ch-index', 
                'silhouette', 'edge-index', 'min-max-cut', 'gap']
    metrics = ['gap']
    exe_met = ExecMetric(pre_p.experiment, metrics)

    for algorithm in exec_alg.algorithms:
        exe_met.run(algorithm, k_min=k_min, k_max=k_max, n_sim=n_sim)
    ############################

    # plot = Plot(pre_p.experiment, exec_alg.algorithms,
    #             exe_met.metrics, k_min=4, k_max=4)
    # plot.plot_data_distribution()
    # plot.plot_k_range()
    # plot.gen_corr_df()
    # plot.plot_intra_inter()
    # plot.cvs_k_range()


    ############################
