import numpy as np
from sklearn.cluster import MiniBatchKMeans as mbm
from sklearn.cluster import SpectralClustering as sc

__author__ = "Deepansh Srivastava"


def component_cluster(
    csdm,
    *args,
    method="spectral",
    norm=False,
    min_threshold=0.05,
    n_steps=1,
    n_repeats=1,
    **kwargs
):
    """Spectral clustering"""
    input_sum = csdm.max()
    csdm_norm = csdm / csdm.max()

    assert csdm.x[0].coordinates.unit == csdm.x[1].coordinates.unit
    assert csdm.x[0].coordinates.unit == csdm.x[2].coordinates.unit

    if norm:
        mean_vec = csdm_norm.mean(axis=2)
        zero_indexes = np.where(mean_vec.y[0].components[0] == 0)
        mean_vec.y[0].components[0][zero_indexes] = 1e12
        csdm_norm.y[0].components[0] /= mean_vec.y[0].components[0][None, :, :]
        csdm_norm /= csdm_norm.max()

    if method == "spectral":
        obj = sc
    elif method == "mini_batch_kmean":
        obj = mbm
    else:
        obj = method

    csdm_zero = csdm_norm.copy()
    csdm_zero.y[0].components[:] = 0.0
    clusters = [csdm_zero.copy() for _ in range(kwargs["n_clusters"])]

    th_add = (1 / n_steps) / n_repeats

    for rep in range(n_repeats):
        indexes = []
        coords = []

        threshold_arrray = np.arange(n_steps) / n_steps + min_threshold + rep * th_add
        th_index = np.where(threshold_arrray > 1)[0]
        if th_index.size != 0:
            threshold_arrray = threshold_arrray[: th_index[0]]

        for th in threshold_arrray:
            indexes.append(np.where(csdm_norm.y[0].components[0] > th))
            next_coords = np.array(
                [
                    dim.coordinates[item].value
                    for dim, item in zip(csdm_norm.x[::-1], indexes[-1])
                ]
            ).T.ravel()
            coords = np.concatenate((coords, next_coords))

        coords = np.array(coords).reshape(-1, 3)

        clustering_result = obj(*args, **kwargs).fit(coords)
        label_counts = int(clustering_result.labels_.max()) + 1

        for lbl_index in range(label_counts):
            mask = np.zeros(csdm_norm.y[0].components[0].shape)
            labels = clustering_result.labels_
            start_lbls = 0
            for idxs in indexes:
                lbls = labels[start_lbls : idxs[0].size + start_lbls]
                cl_idx = np.where(lbls == lbl_index)
                mask[idxs[0][cl_idx], idxs[1][cl_idx], idxs[2][cl_idx]] += 1
                start_lbls += idxs[0].size
            cl_csdm = csdm_norm.copy()
            if n_steps > 1:
                cl_csdm.y[0].components[0] = mask
            else:
                cl_csdm.y[0].components[0] *= mask
            clusters[lbl_index] += cl_csdm

    factor = input_sum / np.sum(clusters).max()
    clusters = [item * factor for item in clusters]
    return clusters, clustering_result
