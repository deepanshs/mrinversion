from .nmr import kernel


class NMR:
    def __new__(self, spectrum, points, extent, oversampling=2, **kwargs):
        return kernel(spectrum, points, extent, oversampling, **kwargs)


class Relaxation:
    pass
