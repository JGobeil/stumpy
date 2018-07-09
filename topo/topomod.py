import numpy as np
from scipy import ndimage
from functools import wraps

from .topo import Topo
from ..helper import lazy_property


class _TopoScipyNdImage(Topo):
    _ndimagefunc = None

    def __init__(self, src, **params):
        super().__init__(src)
        self.params = params

    @property
    def data(self):
        return self.__class__._ndimagefunc(self.src.data, **self.params)

    def __str__(self):
        return self.name + str(self.params)

class GaussianFilter(_TopoScipyNdImage):
    """ Gaussian filter of a Topo """
    _ndimagefunc = ndimage.gaussian_filter


class GaussianLaplace(_TopoScipyNdImage):
    _ndimagefunc = ndimage.gaussian_laplace


class GreyErosion(_TopoScipyNdImage):
    _ndimagefunc = ndimage.grey_erosion


class BinaryErosion(_TopoScipyNdImage):
    _ndimagefunc = ndimage.binary_erosion


class GreyDilation(_TopoScipyNdImage):
    _ndimagefunc = ndimage.grey_dilation


class BinaryDilation(_TopoScipyNdImage):
    _ndimagefunc = ndimage.binary_dilation


class GreyClosing(_TopoScipyNdImage):
    _ndimagefunc = ndimage.grey_closing


class GreyOpening(_TopoScipyNdImage):
    _ndimagefunc = ndimage.grey_opening


class BinaryClosing(_TopoScipyNdImage):
    _ndimagefunc = ndimage.binary_closing


class Floor(Topo):
    def __init__(self, limit, src, data_limit=None):
        super().__init__(src)

        self.limit = limit
        if data_limit is None:
            self.data_limit = src
        else:
            self.data_limit = data_limit

    @property
    def data(self):
        src = self.src
        srcdata = src.data

        zmin = np.min(srcdata)
        zmax = np.max(srcdata)

        zlimit = zmin + (zmax - zmin)*self.limit
        data = self.src.data - zlimit
        data[data < 0] = 0

        return data


class Ceil(Topo):
    def __init__(self, limit, src, data_limit=None):
        super().__init__(src)

        self.limit = limit
        if data_limit is None:
            self.data_limit = src
        else:
            self.data_limit = data_limit

    @property
    def data(self):
        src = self.src
        srcdata = src.data

        zmin = np.min(srcdata)
        zmax = np.max(srcdata)

        zlimit = zmax - (zmax - zmin)*self.limit
        data = self.src.data - zlimit
        data[data > 0] = 0

        return data


class Binary(Topo):
    def __init__(self, src, limit=0.1, limit_src=None, value=(0, 1)):
        super().__init__(src.channel_number, src.sxm)
        self.src = src
        self.limit = limit
        self.limit_src = limit_src
        self.value = value

    @property
    def data(self):
        if self.limit_src is None:
            src = self.src
        else:
            src = self.limit_src

        data = self.src.data

        zmin = np.min(data)
        zmax = np.max(data)

        zlimit = zmin + (zmax - zmin)*self.limit

        cond = data < zlimit

        data[cond] = self.value[0]
        data[np.logical_not(cond)] = self.value[1]

        return data


class Multiply(Topo):
    def __init__(self, src, mult):
        super().__init__(src.channel_number, src.sxm)
        self.src = src
        self.mult = mult

    @property
    def data(self):
        if isinstance(self.mult, Topo):
            return self.src.data * self.mult.data
        else:
            return self.src.data * self.mult
