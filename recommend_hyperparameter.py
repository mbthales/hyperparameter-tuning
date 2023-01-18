import pyreadr
from pathlib import Path
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def get_hyperparameters(path_file):
  if Path(path_file).suffix == ".RData":
    return pyreadr.read_r(path_file)[Path(path_file).stem]
  elif Path(path_file).suffix == ".csv":
    return pd.read_csv(path_file, index_col=False)  

def normalize_hyperparameters(method, hp):
  if method == "minmax":
    scaler = MinMaxScaler()
    scaler.fit(hp)
    return pd.DataFrame(scaler.transform(hp), index=hp.index, columns=hp.columns)
  elif method == "zscore":
    return zscore(hp)
  
def get_hyperparameters_clustering_labels(method, num_of_clusters, n_hp):
  if method == "kmeans":
    return KMeans(n_clusters = num_of_clusters, n_init = 10).fit_predict(n_hp)
  elif method == "s_clustering":
    return SpectralClustering(n_clusters = num_of_clusters, n_init = 10).fit_predict(n_hp)

def cluster_hyperparameters(n_hp, labels):
  n_hp.loc[:,"cluster"] = labels
  return n_hp

def pick_candidate_cluster(labels, clustering_hp):
  unique_labels = set(labels)
  lowest_mse_average = 0
  cluster_with_lowest_mse_average = 0
  for label in unique_labels:
    average_mse_cluster = clustering_hp.loc[clustering_hp['cluster'] == label]["mse"].mean()
    if average_mse_cluster < lowest_mse_average:
      lowest_mse_average = average_mse_cluster
      cluster_with_lowest_mse_average = label
  return clustering_hp.loc[clustering_hp['cluster'] == cluster_with_lowest_mse_average]

def recomendation_in_candidate_cluster(candidate_cluster):
  return candidate_cluster[candidate_cluster["mse"] == candidate_cluster["mse"].min()].index

def get_recommend_hyperparameter(path_file, normalization_method, clustering_method, num_of_clusters):
  hp = get_hyperparameters(path_file)
  normalized_hp = normalize_hyperparameters(normalization_method, hp)
  clustering_labels = get_hyperparameters_clustering_labels(clustering_method, num_of_clusters, normalized_hp)
  clustering_hp = cluster_hyperparameters(normalized_hp, clustering_labels)
  candidate_cluster = pick_candidate_cluster(clustering_labels, clustering_hp)
  recomendation_candidate_cluster = recomendation_in_candidate_cluster(candidate_cluster)
  return hp.iloc[recomendation_candidate_cluster]

print(get_recommend_hyperparameter(path_file="hyperparameters.csv",normalization_method="minmax", clustering_method="kmeans", num_of_clusters=8))
