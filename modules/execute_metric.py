import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from algorithms.clustering.metrics import Metrics
from sklearn.metrics import silhouette_score, silhouette_samples

class ExecMetric():

    def __init__(self, experiment, metrics):
        self.experiment = experiment
        self.metrics = metrics

    @property
    def exp_path(self):
        return f"booking/{self.experiment}"

    def run(self, algorithm, k_min=2, k_max=2, n_sim=30):

        alg_name = algorithm.__str__()
        data_file = pd.read_csv(f"{self.exp_path}/data.csv")
        
        k_range = range(k_min, k_max+1)
        
        all_metrics_df = pd.DataFrame()
        for k in tqdm(k_range, desc=f"k:"):
            metric_df = pd.DataFrame()
            for met in self.metrics:
                met_results = []
                if met == 'silhouette':
                    s_sil_df = pd.DataFrame()
                for n in range(n_sim):
                    clusters_labels = pd.read_csv(
                        f"{self.exp_path}/{alg_name}/{k}/{algorithm.__str__()}_k{k}_clusters_labels.csv")
                    
                    n_sim_labels = clusters_labels[f'sim_{n}']
                    
                    centroids = self.get_centroids(n_sim_labels, data_file, k)
                    if met == 'silhouette':
                        met_results.append(silhouette_score(data_file.drop(columns='labels').values, 
                                                n_sim_labels))
                        sample_silhouette_values = silhouette_samples(
                            data_file.drop(columns='labels').values, n_sim_labels)
                        s_sil_df[f'silhouette_samples_sim_{n}'] = sample_silhouette_values 
                        s_sil_df[f'labels_sim_{n}'] = n_sim_labels
                    else:
                        met_results.append(Metrics.evaluate(met, 
                                        data_file.drop(columns='labels').values, 
                                        centroids, 
                                        algorithm=algorithm))
        
                metric_df[f'{met}'] = met_results
                if met == 'silhouette':
                    s_sil_df.to_csv(f"{self.exp_path}/{alg_name}/{k}/{alg_name}_k{k}_silhouette_samples.csv", index=False) 

            metric_df = metric_df.assign(k=k)
            metric_df = metric_df.assign(algorithm=alg_name)
            metrics_path = f"{self.exp_path}/{alg_name}/{k}/{alg_name}_k{k}_metrics_results.csv"
            metric_df.to_csv( metrics_path, index=False)
            all_metrics_df = pd.concat([all_metrics_df, metric_df])
        all_metrics_df.to_csv(f'{self.exp_path}/{alg_name}/{alg_name}_all_metrics_results.csv', index=False)
                
    def run_from_sklearn(self, linkage, k_min=2, k_max=2):
        alg_name = f'AGL-{linkage}'
        data_file = pd.read_csv(f"{self.exp_path}/data.csv")
        
        k_range = range(k_min, k_max+1)
        
        all_metrics_df = pd.DataFrame()
        for k in tqdm(k_range, desc=f"k:"):
            metric_df = pd.DataFrame()
            for met in self.metrics:
                
                clusters_labels = pd.read_csv(
                    f"{self.exp_path}/{alg_name}/{k}/{alg_name}_k{k}_clusters_labels.csv")
                
                n_sim_labels = clusters_labels[f'sim_{0}']
                
                centroids = self.get_centroids(n_sim_labels, data_file, k)
                if met == 'silhouette':
                    result = silhouette_score(data_file.drop(columns='labels').values, 
                                                n_sim_labels)
                    sample_silhouette_values = silhouette_samples(
                        data_file.drop(columns='labels').values, n_sim_labels)
                    s_sil_df = pd.DataFrame()
                    s_sil_df['silhouette_samples_sim_0'] = sample_silhouette_values 
                    s_sil_df['labels_sim_0'] = n_sim_labels
                    s_sil_df.to_csv(f"{self.exp_path}/{alg_name}/{k}/{alg_name}_k{k}_silhouette_samples.csv", index=False) 
                else:
                    result = Metrics.evaluate(met, data_file.drop(columns='labels').values, 
                                        centroids)
                metric_df[f'{met}'] = [result]

            metric_df = metric_df.assign(k=k)
            metric_df = metric_df.assign(algorithm=alg_name)
            metrics_path = f"{self.exp_path}/{alg_name}/{k}/{alg_name}_k{k}_metrics_results.csv"
            metric_df.to_csv( metrics_path, index=False)
            all_metrics_df = pd.concat([all_metrics_df, metric_df])
        all_metrics_df.to_csv(f'{self.exp_path}/{alg_name}/{alg_name}_all_metrics_results.csv', index=False)
    

    def get_centroids(self, labels, data, n_clusters):
        data['labels'] = labels
        centroids = []
        for k in range(n_clusters):
            cluster = data.loc[data.labels == k].drop(columns='labels')
            centroids.append(np.mean(cluster.values, axis=0))
        return centroids