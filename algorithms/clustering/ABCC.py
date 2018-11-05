import numpy as np
from ..optimization.ABC import ABC
from .metrics import Metrics


class ABCCObjectiveFunction:
    def __init__(self, data, k, n_attributes, metric_type, evaluation_metric):
        self.function = self.evaluate
        self.minf = 0.0
        self.maxf = 1.0
        self.k = k
        self.n_attributes = n_attributes
        self.data = data
        self.metric_type = metric_type
        self.evaluation_metric = evaluation_metric

    def evaluate(self, x):
        centroids = x.reshape((self.k, self.n_attributes))
        clusters = {}

        for k in range(self.k):
            clusters[k] = []

        for xi in self.data:
            # dist = [(np.linalg.norm(xi - centroids[c])**2) for c in range(len(centroids))]
            dist = [squared_euclidean_dist(xi, centroids[c])
                    for c in range(len(centroids))]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        return Metrics.evaluate(self.evaluation_metric, self.data, centroids, clusters)
        # return self.evaluation_metric(data=self.data, centroids=centroids, clusters=clusters)


# def sse(centroids, clusters):
#     global_intra_cluster_sum = 0.0

#     for c in range(len(centroids)):
#         partial_intra_cluster_sum = 0.0

#         if len(clusters[c]) > 0:
#             for point in clusters[c]:
#                 partial_intra_cluster_sum += (
#                     squared_euclidean_dist(point, centroids[c]))

#         global_intra_cluster_sum += partial_intra_cluster_sum

#     return global_intra_cluster_sum


def squared_euclidean_dist(u, v):
    sed = ((u - v) ** 2).sum()
    return sed


class ABCC(object):
    def __init__(self, k=2, swarm_size=50, n_iter=20, trials_limit=100, metric_type='min',evaluation_metric=None):
        self.k = k
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.trials_limit = trials_limit
        self.evaluation_metric = evaluation_metric
        self.metric_type = metric_type

    def fit(self, data, k):
        self.k = k

        self.n_attributes = data.shape[1]

        self.abc = ABC(ABCCObjectiveFunction(data, self.k, self.n_attributes, self.metric_type, self.evaluation_metric),
                       dim=self.k * self.n_attributes,
                       colony_size=self.swarm_size,
                       n_iter=self.n_iter,
                       trials_limit=self.trials_limit)

        self.abc.optimize()

        self.centroids = {}
        raw_centroids = self.abc.gbest.pos.reshape(
            (self.k, self.n_attributes))

        self.convergence = self.abc.optimum_cost_tracking_iter

        for c in range(len(raw_centroids)):
            self.centroids[c] = raw_centroids[c]

        self.clusters = self.get_clusters(self.centroids, data)

        self.number_of_effective_clusters = 0

        for c in range(len(self.centroids)):
            if len(self.clusters[c]) > 0:
                self.number_of_effective_clusters = self.number_of_effective_clusters + 1

    @staticmethod
    def get_clusters(centroids, data):

        clusters = {}
        for c in centroids:
            clusters[c] = []

        for xi in data:
            dist = [np.linalg.norm(xi - centroids[c]) for c in centroids]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        return clusters
    
    def predict(self, x):
        if len(x.shape) > 1:
            class_ = []
            for c in self.centroids:
                class_.append(np.sum((x - self.centroids[c]) ** 2, axis=1))
            return np.argmin(np.array(class_).T, axis=1)
        else:
            dist = [np.linalg.norm(x - self.centroids[c])
                    for c in self.centroids]
            class_ = dist.index(min(dist))
            return class_

    def __str__(self):
        return f"ABCC-{self.evaluation_metric}"
