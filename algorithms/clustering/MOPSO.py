# def evaluate(self, x):
#         centroids = x.reshape((self.k, self.n_attributes))
#         clusters = {}

#         for k in range(self.k):
#             clusters[k] = []

#         for xi in self.data:
#             # dist = [(np.linalg.norm(xi - centroids[c])**2) for c in range(len(centroids))]
#             dist = [squared_euclidean_dist(xi, centroids[c])
#                     for c in range(len(centroids))]
#             class_ = dist.index(min(dist))
#             clusters[class_].append(xi)

#         return Metrics.evaluate(self.evaluation_metric, self.data, centroids, clusters)

from platypus import OMOPSO, NSGAII, Problem, Real
from algorithms.clustering.metrics import Metrics
import numpy as np
import pandas as pd


def intra_inter(x):
    k = 3
    features = 21
    data = pd.read_csv('booking/acs/data.csv')
    x = np.array(x)
    centroids = x.reshape((k, features))
    return [
        Metrics.evaluate('intra-cluster', data.values, centroids),
        Metrics.evaluate('inter-cluster', data.values, centroids)
        ]

k = 3
features = 21
problem = Problem(features*k, 2)
problem.types[:] = Real(0, 1)
problem.function = intra_inter
problem.directions[:1] = Problem.MAXIMIZE
algorithm = OMOPSO(problem, epsilons= [0.05])
algorithm.run(100)
for solution in algorithm.archive._contents:
    print(solution.variables)

