import random
from typing import List

import numpy as np
import torch
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from sklearn.metrics import pairwise_distances, silhouette_samples
from tqdm import tqdm

pca = PCA(n_components=24)

def gmm_monitor(
        model,
        train_data_spikes: Tensor,
        train_data_labels: Tensor,
        test_data_spikes: Tensor,
        test_data_labels: Tensor,
        test_labels: List[int],
        device='cuda',
        epochs=1,
        use_pca=False,
        use_scaler=False,
        n_init=1,
        max_iter=100,
        verbose=False,
        covariance_type='tied',
        use_iso=True,
        score=True,
        test_data_origin=None,
):
    train_data_spikes, test_data_spikes, test_data_labels, train_data_labels, classes = _data_process(model,
                                                                             train_data_spikes,
                                                                             train_data_labels,
                                                                             test_data_spikes,
                                                                             test_data_labels,
                                                                             test_labels,
                                                                             device=device,
                                                                             use_pca=use_pca,
                                                                             use_scaler=use_scaler, use_iso=use_iso,
                                                                                                      test_data_origin=test_data_origin,)

    scores = []
    gmm_tests = []
    for i in range(epochs):
        gmm = GaussianMixture(
            classes,
            # random_state=random.randint(0, 114514),
            random_state=i,
            covariance_type=covariance_type,
            max_iter=max_iter,
            verbose=verbose,
            n_init=n_init,
            tol=1e-4,
        ).fit(train_data_spikes)
        gmm_test = gmm.predict(test_data_spikes)
        if score:
            score = adjusted_rand_score(test_data_labels, gmm_test)
        else:
            score = 0
        scores.append(score)
        gmm_tests.append(gmm_test)

    return scores, gmm_tests, test_data_spikes

def bgmm_monitor(
        model,
        train_data_spikes: Tensor,
        train_data_labels: Tensor,
        test_data_spikes: Tensor,
        test_data_labels: Tensor,
        test_labels: List[int],
        device='cuda',
        epochs=1,
        use_pca=False,
        use_scaler=False,
        n_init=1,
        max_iter=100,
        verbose=False,
        covariance_type='tied',
        use_iso=True,
        score=True,
):
    train_data_spikes, test_data_spikes, test_data_labels, train_data_labels, classes = _data_process(model,
                                                                             train_data_spikes,
                                                                             train_data_labels,
                                                                             test_data_spikes,
                                                                             test_data_labels,
                                                                             test_labels,
                                                                             device=device,
                                                                             use_pca=use_pca,
                                                                             use_scaler=use_scaler, use_iso=use_iso)

    scores = []
    gmm_tests = []
    for i in range(epochs):
        gmm = BayesianGaussianMixture(
            n_components=classes,
            # random_state=random.randint(0, 114514),
            max_iter=max_iter,
            weight_concentration_prior=1e-2,
            random_state=i,
            verbose=verbose,
            n_init=n_init,
            weight_concentration_prior_type='dirichlet_process'
        ).fit(train_data_spikes)
        gmm_test = gmm.predict(test_data_spikes)
        if score:
            score = adjusted_rand_score(test_data_labels, gmm_test)
        else:
            score = 0
        scores.append(score)
        gmm_tests.append(gmm_test)

    return scores, gmm_tests, test_data_spikes

def kmeans_monitor(
        model,
        train_data_spikes: Tensor,
        train_data_labels: Tensor,
        test_data_spikes: Tensor,
        test_data_labels: Tensor,
        test_labels: List[int],
        device='cuda',
        epochs=1,
        use_iso=True,
):
    train_data_spikes, test_data_spikes, test_data_labels, train_data_labels, classes = _data_process(model,
                                                                             train_data_spikes,
                                                                             train_data_labels,
                                                                             test_data_spikes,
                                                                             test_data_labels,
                                                                             test_labels,
                                                                             device=device,
                                                                             use_iso=use_iso)

    scores = []
    kmeans_tests =[]
    for i in range(epochs):
        kmeans = KMeans(
            n_clusters=classes,
            random_state=i,
        )
        kmeans.fit(train_data_spikes)
        predictions = kmeans.predict(test_data_spikes)
        # score = adjusted_rand_score(test_data_labels.cpu().numpy(), predictions)
        score = 0
        scores.append(score)
        kmeans_tests.append(predictions)

    return scores, kmeans_tests, test_data_spikes

def knn_monitor(
        model,
        train_data_spikes: Tensor,
        train_data_labels: Tensor,
        test_data_spikes: Tensor,
        test_data_labels: Tensor,
        test_labels: List[int],
        device='cuda',
        k=100,
        epochs=1,
):
    train_data_spikes, test_data_spikes, test_data_labels, train_data_labels, classes = _data_process(model,
                                                                             train_data_spikes,
                                                                             train_data_labels,
                                                                             test_data_spikes,
                                                                             test_data_labels,
                                                                             test_labels,
                                                                             device=device,)

    scores = []
    for i in range(epochs):
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_data_spikes, train_data_labels)
        predictions = knn.predict(test_data_spikes)
        acc = accuracy_score(test_data_labels, predictions)
        scores.append(acc)
    return scores, predictions, test_data_spikes

def _data_process(
        model,
        train_data_spikes: Tensor,
        train_data_labels: Tensor,
        test_data_spikes: Tensor,
        test_data_labels: Tensor,
        test_labels: List[int],
        device='cuda',
        use_pca=False,
        use_scaler=True,
        use_iso=True,
        test_data_origin=None,
):
    # np.random.seed(42)
    # random.seed(42)

    # mask = torch.isin(test_data_labels, torch.tensor(test_labels))
    # indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    #
    # test_data_spikes = test_data_spikes[indices].to(device)
    # test_data_labels = test_data_labels[indices]
    # test_data_labels = test_data_labels.to(torch.int64)
    #
    # mask = torch.isin(train_data_labels, torch.tensor(test_labels))
    # indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    #
    # train_data_spikes = train_data_spikes[indices].to(device)
    # train_data_labels = train_data_labels[indices]
    # train_data_labels = train_data_labels.to(torch.int64)

    if train_data_spikes is not None:
        train_data_spikes = torch.stack([_zscore_normalize(spike) for spike in train_data_spikes])

    # test_data_spikes = torch.stack([_zscore_normalize(spike) for spike in test_data_spikes])

    if model:
        with torch.no_grad():
            test_data_spikes = inference_in_chunks(model, test_data_spikes, test_data_origin=test_data_origin)
            # model.eval()
            # test_data_spikes = model.transform(test_data_spikes.to(device)).cpu().numpy()
            if train_data_spikes is not None:
                train_data_spikes = model.transform(train_data_spikes).cpu().numpy()
            # print("embedding std:", np.std(train_data_spikes))
            # print("embedding mean:", np.mean(train_data_spikes))

    if use_pca:
        if train_data_spikes is not None:
            train_data_spikes = pca.fit_transform(train_data_spikes)
        test_data_spikes = pca.fit_transform(test_data_spikes)

    if use_scaler:
        if train_data_spikes is not None:
            scaler = StandardScaler()
            scaler.fit(train_data_spikes)
            train_data_spikes = scaler.transform(train_data_spikes)
            test_data_spikes = scaler.transform(test_data_spikes)
        else:
            scaler = StandardScaler()
            test_data_spikes = scaler.fit_transform(test_data_spikes)

    if use_iso:
        iso = IsolationForest(max_samples=1024, contamination=0.15, random_state=42)
        pred = iso.fit_predict(test_data_spikes)
        train_data_spikes = test_data_spikes[pred==1]
        print(train_data_spikes.shape)
        print(test_data_spikes.shape)

    if train_data_spikes is not None:
        classes = min(np.unique(train_data_labels).size, np.unique(test_data_labels).size)
        return train_data_spikes, test_data_spikes, test_data_labels, train_data_labels, classes
    else:
        classes = np.unique(test_data_labels).size
        return test_data_spikes, test_data_spikes, test_data_labels, test_data_labels, classes

def _zscore_normalize(spike, dim=0):
    std = spike.std(dim=dim, keepdim=True) + 1e-6
    mean = spike.mean(dim=dim, keepdim=True)
    # return (spike - mean) / std
    return spike

def inference_in_chunks(model, test_data_spikes, batch_size=4096 * 2, device='cuda', test_data_origin=None):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, test_data_spikes.shape[0], batch_size):
            batch = test_data_spikes[i : i + batch_size].to(device)
            if test_data_origin is not None:
                batch2 = test_data_origin[i : i + batch_size].to(device)
                output = model.transform(batch2, batch)
            else:
                output = model.transform(batch, None)
            outputs.append(output.cpu())

    return torch.cat(outputs, dim=0)

# def evaluate_clusters(X, labels, silhouette_threshold=0.2, distance_std_threshold=0.5):
#     """
#     对每个 cluster 给出是否值得进一步细分的建议。
#
#     参数:
#     - X: 特征矩阵 [n_samples, n_features]
#     - labels: 聚类标签
#     - silhouette_threshold: 轮廓系数低于该值则考虑划分
#     - distance_std_threshold: 类内距离的 std 高于该值则考虑划分
#
#     返回:
#     - problematic_clusters: list of cluster ids 值得进一步划分的聚类编号
#     - metrics: 每个聚类的 silhouette_mean, intra_dist_mean, intra_dist_std
#     """
#     unique_labels = np.unique(labels)
#     # silhouette_vals = silhouette_samples(X, labels)
#     silhouette_vals = 0
#
#     problematic_clusters = []
#     cluster_metrics = {}
#
#     for lbl in unique_labels:
#         idx = (labels == lbl)
#         X_cluster = X[idx]
#
#         if len(X_cluster) <= 2:
#             continue  # 点太少无法判断
#
#         # 类内距离矩阵
#         dists = _compute_intra_cluster_distances(X_cluster)
#         intra_dists = dists[np.triu_indices_from(dists, k=1)]
#         dist_mean = np.mean(intra_dists)
#         dist_std = np.std(intra_dists)
#
#         # 轮廓系数
#         sil_vals = silhouette_vals[idx]
#         sil_mean = np.mean(sil_vals)
#
#         cluster_metrics[lbl] = {
#             "silhouette_mean": sil_mean,
#             "intra_dist_mean": dist_mean,
#             "intra_dist_std": dist_std,
#         }
#
#         if sil_mean < silhouette_threshold or dist_std > distance_std_threshold:
#             problematic_clusters.append(lbl)
#
#     return problematic_clusters, cluster_metrics

def _compute_intra_cluster_distances(X_cluster, max_n=1000):
    n = len(X_cluster)
    if n > max_n:
        idx = np.random.choice(n, size=max_n, replace=False)
        X_sample = X_cluster[idx]
    else:
        X_sample = X_cluster

    dists = pairwise_distances(X_sample)
    return dists[np.triu_indices_from(dists, k=1)]
