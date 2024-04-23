from utils.faiss_rerank import compute_jaccard_distance
from sklearn.cluster import DBSCAN
import torch

a = torch.randn(33,256)
rerank_dist = compute_jaccard_distance(a, k1=20, k2=6)
cluster = DBSCAN(eps=0.6, min_samples=2, metric='precomputed', n_jobs=-1)
pseudo_labels = cluster.fit_predict(rerank_dist)
# print(pseudo_labels)
