import numpy as np

# from mrinversion.util import _check_dimension_type


def T2(direct_dimension, inverse_dimension):
    # _check_dimension_type(direct_dimension, inverse_dimension)
    x = direct_dimension.coordinates
    x_inverse = inverse_dimension.coordinates
    return np.exp(np.tensordot(-x, (1 / x_inverse), 0))


def T2_old(x, nx=2, rangex=[0, 1], oversample=1, log_scale=True):

    if rangex[0] < 0 or rangex[1] < 0:
        raise ValueError("Range cannot include negative values.")

    if log_scale:
        x_min = np.log10(rangex[0])
        x_max = np.log10(rangex[1])
    else:
        x_min = rangex[0]
        x_max = rangex[1]

    m = x.size
    xinv = ((np.arange(nx * oversample)) / (nx * oversample - 1)) * (x_max - x_min)
    xinv += x_min

    if log_scale:
        xinv = 10 ** (xinv)

    # print (xinv)

    k = np.exp(np.tensordot(-x, (1 / xinv), 0))
    # print (K.shape)
    k = k.ravel().reshape(m, nx, oversample).sum(axis=2)
    # print (K.shape)
    xinv = ((np.arange(nx)) / (nx - 1)) * (x_max - x_min) + x_min
    # print ('log?', xinv)
    return xinv, k
