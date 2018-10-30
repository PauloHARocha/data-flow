import numpy as np


class KMeans:

    def __init__(self, k=4, n_iter=300, tol=0.00001, data=None):
        self.k = k  # Number of centroids
        self.n_iter = n_iter  # Number of iterations
        self.tol = tol  # Tolerance
        self.data = data  # Data input, can be called X
        self.centroids = {}  # Position of centroids
        self.clusters = {}  # Clusters
        self.all_centroids = {}
        self.all_clusters = {}

    # Initialize the centroids randomly based on the position of the data
    def init_centroids(self):
        # r = np.random.permutation(self.data.shape[0])
        # for k in range(self.k):
        #     self.centroids[k] = self.data[r[k]]
        for k in range(self.k):
            self.centroids[k] = np.random.random(self.data.shape[1])

    def fit(self, data=None, k=None):

        if data is not None:
            self.data = data

        if k is not None:
            self.k = k
        
        self.init_centroids()

        for itr in range(self.n_iter):
            self.clusters = {}  # initialize the clusters
            for k in range(self.k):
                self.clusters[k] = []  # For each k, create an array

            for xi in self.data:  # Atribute each data on dataset to one of the centroids, based on the distance
                # calculate the distance from data to each centroid
                dist = [np.linalg.norm(xi - self.centroids[c])
                        for c in self.centroids]
                # get the centroid index that has the minimun distance from data
                class_ = dist.index(min(dist))
                # append the data to the cluster of the centroid
                self.clusters[class_].append(xi)
            
            self.all_centroids[itr] = dict(self.centroids)
            
            self.all_clusters[itr] = self.clusters

            old_centroids = dict(self.centroids)
            for k in self.clusters:  # Update the position of each centroid based on the average distance from each data of the centroid
                if len(self.clusters[k]) > 0:
                    self.centroids[k] = np.average(self.clusters[k], axis=0)
         

            is_done = True
            for k in self.centroids:  # If the centroid doesnt change position end the run
                old_centroid = old_centroids[k]
                centroid = self.centroids[k]
                if np.linalg.norm(old_centroid - centroid) > self.tol:
                    is_done = False
            
            if is_done:
                break

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
        return 'K-means'