import numpy as np
from scipy import ndimage

from .topo import Topo


class _TopoScipyNdImage(Topo):
    _ndimagefunc = None

    def __init__(self, src, listpos=None, **params):
        super().__init__(src, listpos=listpos)
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
    def __init__(self, src, limit, data_limit=None, value=0.0, **kwargs):
        super().__init__(src, **kwargs)

        self.limit = limit
        if data_limit is None:
            self.data_limit = src
        else:
            self.data_limit = data_limit
        self.value = value

    @property
    def data(self):
        data = self.src.data

        zmin = np.min(data)
        zmax = np.max(data)

        zlimit = zmin + (zmax - zmin)*self.limit
        data[(data - zlimit) < 0] = 0

        return data


class Ceil(Topo):
    def __init__(self, src, limit, data_limit=None, value=0.0, **kwargs):
        super().__init__(src, **kwargs)

        self.limit = limit
        if data_limit is None:
            self.data_limit = src
        else:
            self.data_limit = data_limit
        self.value = value

    @property
    def data(self):
        data = self.src.data

        zmin = np.min(data)
        zmax = np.max(data)

        zlimit = zmax - (zmax - zmin)*self.limit
        data[(data - zlimit) > 0] = value

        return data


class Binary(Topo):
	def __init__(self, src, limit=0.1, value=(0, 1), **kwargs):
		super().__init__(src, **kwargs)
		self.limit = limit
		self.value = value

	@property
	def data(self):
		data = self.src.data

		zmin = np.min(data)
		zmax = np.max(data)

		zlimit = zmin + (zmax - zmin)*self.limit
		print(zlimit)

		cond = data < zlimit

		data[cond] = self.value[0]
		data[np.logical_not(cond)] = self.value[1]

		return data


class CorrectTipChange(Topo):
    def __init__(self, src, toll=None, corr_factor=1, **kwargs):
        super().__init__(src, **kwargs)
        if toll is not None:
            self.toll = toll
        else:
            self.toll = 1.5e-11

        self.corr_factor = corr_factor

    @property
    def lines_diff(self):
        data = super().data.T
        N = data.shape[0] - 1
        diff = np.empty(N)
        for i in range(0, N):
            diff[i] = np.mean(data[i + 1, :] - data[i, :])
        return diff

    @property
    def data(self):
        data = super().data.T

        diff = self.lines_diff

        diff[np.abs(diff) < self.toll] = 0
        cumdiff = np.cumsum(np.insert(diff*self.corr_factor, 0, 0))

        return (data.T - cumdiff)

    def plot_lines_diff(self, ax=None, size=None, pyplot=True):
        if ax is None:
            figure = create_figure(size=size, pyplot=pyplot)
            ax = figure.add_subplot(111)
        ax.plot(np.abs(self.lines_diff))
        ax.axhline(self.toll, color='red')
        return ax

class DriftCorrection(Topo):
    def __init__(self, src, order=1, **kwargs):
        super().__init__(src, **kwargs)
        self.order = order

    @property
    def data(self):
        data, means, fits, lines = self.get_corrected_data()
        return data

    def get_corrected_data(self):
        data = super().data
        means = np.mean(data, axis=0)
        lines = np.arange(means.size)
        fits = np.poly1d(np.polyfit(lines, means, self.order))(lines)
        corrdata = data - fits
        return corrdata, means, fits, lines

    def plot_fit(self, ax=None, size='small', pyplot=True):
        if ax is None:
            figure = create_figure(size=size, pyplot=pyplot)
            ax = figure.add_subplot(111)
        data, means, fits, lines = self.get_corrected_data()
        ax.plot(lines, means)
        ax.plot(lines, fits)
        return ax


class SubstractAverage(Topo):
    @property
    def data(self):
        data = self.src.data
        return data - np.mean(data, axis=0)


class Multiply(Topo):
    def __init__(self, src, other, **kwargs):
        super().__init__(src, **kwargs)
        self.other = other

    @property
    def data(self):
        data1 = self.src.data
        if self.isinlist:
            try:
                data2 = self.other[self.listpos].data
            except TypeError:
                data2 = self.other.data
        else:
            data2 = self.other.data
        return data1 * data2

class Substract(Topo):
    def __init__(self, src, other, **kwargs):
        super().__init__(src, **kwargs)
        self.other = other

    @property
    def data(self):
        data1 = self.src.data
        if self.isinlist:
            try:
                data2 = self.other[self.listpos].data
            except TypeError:
                data2 = self.other.data
        else:
            data2 = self.other.data
        return data1 - data2


class Zeroed(Topo):
    def __init__(self, src, limits, **kwargs):
        super().__init__(src, **kwargs)
        self.limits = limits

    @property
    def data(self):
        data = self.src.data

        x1, x2, y1, y2 = self.limits
        zero = np.mean(data[x1:x2, y1:y2])
        return data - zero
