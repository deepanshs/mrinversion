import numpy as np


def T2(x, nx=2, rangex=[0, 1], oversample=1, log_scale=True):

    message_extension = "Only positive range is defined"
    if rangex[0] < 0:
        message = "ValueError: 'rangex[0]" + message_extension
        print(message)
        return
    if rangex[1] < 0:
        message = "ValueError: 'rangex[1]" + message_extension
        print(message)
        return

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
