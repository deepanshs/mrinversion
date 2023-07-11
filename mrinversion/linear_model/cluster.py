import numpy as np
from csdmpy import statistics as stats
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
    sigma=1,
    **kwargs
):
    """Spectral clustering"""
    input_sum = csdm.max()
    csdm_norm = csdm / csdm.max()

    assert csdm.x[0].coordinates.unit == csdm.x[1].coordinates.unit
    assert csdm.x[0].coordinates.unit == csdm.x[2].coordinates.unit

    inc_coords = [
        np.abs(item.coordinates[1] - item.coordinates[0]).value for item in csdm.x
    ]

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

    n_clusters = (
        kwargs["n_clusters"] if "n_clusters" in kwargs else kwargs["n_components"]
    )
    csdm_zero = csdm_norm.copy()
    csdm_zero.y[0].components[:] = 0.0
    clusters = [csdm_zero.copy() for _ in range(n_clusters)]

    # step_inc = 1 / n_steps
    # th_add = step_inc / n_repeats

    cluster_mean = None
    for _ in range(n_repeats):

        # rep_inc = rep * th_add if n_steps != 1 else 0
        threshold_array = np.linspace(min_threshold, 1, n_steps)  # + rep_inc
        # threshold_arrray = np.append(threshold_arrray, [0.1])

        th_index = np.where(threshold_array > 1)[0]
        if th_index.size != 0:
            threshold_array = threshold_array[: th_index[0]]

        coords, indexes = get_coords_and_indexes(
            csdm_norm, threshold_array, inc_coords, sigma
        )

        clustering_result = obj(*args, **kwargs).fit(coords)
        label_counts = int(clustering_result.labels_.max()) + 1
        labels = clustering_result.labels_

        for lbl_index in range(label_counts):
            mask = get_cluster_mask(csdm_norm, labels, lbl_index, indexes)
            cl_csdm = csdm_norm.copy()
            if n_steps > 1:
                cl_csdm.y[0].components[0] = mask
            else:
                cl_csdm.y[0].components[0] *= mask

            if cluster_mean is None:
                clusters[lbl_index] += cl_csdm
            else:
                cl_means = np.array([st.value for st in stats.mean(cl_csdm)])
                diff = np.sum((cluster_mean - cl_means) ** 2, axis=1)
                index_min = np.argmin(diff)
                clusters[index_min] += cl_csdm

        cluster_mean = np.array(
            [[st.value for st in stats.mean(item)] for item in clusters]
        )
    factor = input_sum / np.sum(clusters).max()
    clusters = [item * factor for item in clusters]
    return clusters, clustering_result


def get_cluster_mask(csdm_norm, labels, lbl_index, indexes):
    mask = np.zeros(csdm_norm.y[0].components[0].shape)
    start_lbls = 0
    for idxs in indexes:
        lbls = labels[start_lbls : idxs[0].size + start_lbls]
        cl_idx = np.where(lbls == lbl_index)
        for cl_idx_ in cl_idx:
            mask[idxs[0][cl_idx_], idxs[1][cl_idx_], idxs[2][cl_idx_]] += 1
        start_lbls += idxs[0].size
    return mask


def get_coords_and_indexes(csdm_norm, threshold_array, inc_coords, sigma):
    indexes = []
    coords = []
    for th in threshold_array:
        indexes.append(np.where(csdm_norm.y[0].components[0] > th))
        indexes_shape = (len(indexes[-1]), len(indexes[-1][0]))
        total_size = np.prod(indexes_shape)
        rand = np.random.normal(loc=[0], scale=1, size=total_size)
        rand = rand.reshape(len(indexes[-1]), len(indexes[-1][0]))
        next_coords = np.array(
            [
                dim.coordinates[item].value / inc
                for inc, dim, item in zip(
                    inc_coords[::-1], csdm_norm.x[::-1], indexes[-1]
                )
            ]
        )
        next_coords += rand * np.asarray(sigma)[::-1][:, None]
        next_coords = next_coords.T.ravel()
        coords = np.concatenate((coords, next_coords))

    coords = np.array(coords).reshape(-1, 3)
    # coords = (coords - coords.mean(axis=0))
    coords /= coords.std(axis=0)

    return coords, indexes


def aff_matrix(X, delta=1e-3, coords=None):
    print(X.shape, X)
    length = X.shape[0]
    adj_mat = np.zeros((length, length), dtype=float)
    for i, crds in enumerate(X):
        adj_mat[i] = np.exp(-delta * (X - crds) ** 2)
    return adj_mat
