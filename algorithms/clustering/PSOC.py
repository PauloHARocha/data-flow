import numpy as np
from ..optimization.PSO import PSO
from .metrics import Metrics


class PSOCObjectiveFunction:
    def __init__(self, data, k, n_attributes, evaluation_metric):
        self.function = self.evaluate
        self.minf = 0.0
        self.maxf = 1.0
        self.k = k
        self.n_attributes = n_attributes
        self.data = data
        self.evaluation_metric = evaluation_metric

    def evaluate(self, x):
        centroids = x.reshape((self.k, self.n_attributes))
        clusters = {}

        for k in range(self.k):
            clusters[k] = []

        for xi in self.data:
            # dist = [(np.linalg.norm(xi - centroids[c])**2) for c in range(len(centroids))]
            dist = [squared_euclidean_dist(xi, centroids[c]) for c in range(len(centroids))]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        return Metrics.evaluate(self.evaluation_metric, self.data, centroids, clusters)


def squared_euclidean_dist(u, v):
    sed = ((u - v) ** 2).sum()
    return sed

class PSOC(object):
    def __init__(self, k=2, swarm_size=50, n_iter=50, lo_w=0.72, up_w=0.9, c1=1.49,
                 c2=1.49, evaluation_metric=None):
        self.k = k
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.evaluation_metric = evaluation_metric
        self.up_w = up_w
        self.lo_w = lo_w
        self.c1 = c1
        self.c2 = c2
        self.v_max = 0.5

    def fit(self, data, k):
        self.k = k

        self.n_attributes = data.shape[1]

        self.pso = PSO(PSOCObjectiveFunction(data, self.k, self.n_attributes, self.evaluation_metric),
                       dim=self.k * self.n_attributes,
                       swarm_size=self.swarm_size,
                       n_iter=self.n_iter,
                       lo_w=self.lo_w, c1=self.c1, c2=self.c2, v_max=self.v_max)

        self.pso.optimize()

        self.centroids = {}
        raw_centroids = self.pso.gbest.pos.reshape((self.k, self.n_attributes))

        self.convergence = self.pso.optimum_cost_tracking_iter

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
        return f"PSOC-{self.evaluation_metric}"