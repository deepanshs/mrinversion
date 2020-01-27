from copy import deepcopy

import numpy as np

from mrinversion.util import _check_csdm_dimension


class BaseModel:
    """Base kernel class."""

    def __init__(self, direct_dimension, inverse_dimensions, n_dir, n_inv):
        kernel = self.__class__.__name__
        message = (
            f"Exactly {n_inv} inverse dimension(s) is/are required for the "
            f"{kernel} kernel."
        )
        if isinstance(inverse_dimensions, list):
            if len(inverse_dimensions) != n_inv:
                raise ValueError(message)
            if n_inv == 1:
                inverse_dimensions = inverse_dimensions[0]
        else:
            if n_inv != 1:
                raise ValueError(message)

        inverse_dimensions = _check_csdm_dimension(
            inverse_dimensions, "inverse_dimensions"
        )

        message = (
            f"Exactly {n_dir} direct dimension(s) is/are required for the "
            f"{kernel} kernel."
        )
        if isinstance(direct_dimension, list):
            if len(direct_dimension) != n_dir:
                raise ValueError(message)
            if n_dir == 1:
                direct_dimension = direct_dimension[0]
        else:
            if n_dir != 1:
                raise ValueError(message)

        direct_dimension = _check_csdm_dimension(direct_dimension, "direct_dimension")

        self.direct_dimension = direct_dimension
        self.inverse_dimensions = inverse_dimensions

    def _averaged_kernel(self, amp, supersampling):
        """Return the kernel by averaging over the supersampled grid cells."""
        shape = ()
        for item in self.inverse_dimensions:
            shape += (item.count, supersampling)
        shape += (self.direct_dimension.count,)

        K = amp.reshape(shape)

        inv_len = len(self.inverse_dimensions)
        axes = tuple([2 * i + 1 for i in range(inv_len)])
        K = K.sum(axis=axes)

        section = [*[0 for i in range(inv_len)], slice(None, None, None)]
        K /= K[section].sum()

        section = [slice(None, None, None) for _ in range(inv_len + 1)]
        for i, item in enumerate(self.inverse_dimensions):
            if item.coordinates_offset == 0:
                section_ = deepcopy(section)
                section_[i] = 0
                K[section_] /= 2.0

        inv_size = np.asarray([item.count for item in self.inverse_dimensions]).prod()
        K = K.reshape(inv_size, self.direct_dimension.count).T

        return K
