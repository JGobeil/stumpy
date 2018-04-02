""" Topo scan viewing and basic information. Modifying method should be in
seperated files and use the decorator @topomod to link it to the class Topo."""

import matplotlib.cm
import numpy as np

from ..helper.lazy import lazy_property

default_cmap = matplotlib.cm.get_cmap('Blues_r')


class Topo:
    """ A class with the Topo 2D data, it size and pos. ."""

    def __init__(self, src):
        """ Initialize a topo. """
        self._src = src

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        self._src = src
        self.reset()

    def reset(self):
        """ Force a recalculation of property on next call."""
        del self.data, self.size_nm, self.pos_nm, self.angle

    @lazy_property
    def size_nm(self):
        """ The size in nanometers as numpy.array."""
        return self.src.sxm.size_nm

    @lazy_property
    def pos_nm(self):
        """ The position in nanometers as numpy.array."""
        return self.src.sxm.pos_nm

    @lazy_property
    def angle(self):
        """ The angle of the topo,"""
        return self.src.sxm.angle

    @property
    def data(self):
        """ The raw data as numpy.array"""
        return self.src.data

    @property
    def size_px(self):
        """ The size in pixels as numpy.array"""
        return np.array(self.data.shape)

    @lazy_property
    def rotation_matrix(self):
        """ Rotation matrix using the angle of the scan."""
        theta = np.radians(self.angle)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    @lazy_property
    def xy(self):
        """ Position in real space of the pixels. """
        xsize_nm, ysize_nm = self.size_nm
        xsize_px, ysize_px = self.size_px

        x = np.linspace(-xsize_nm/2, xsize_nm/2, xsize_px)
        y = np.linspace(ysize_nm/2, -ysize_nm/2, ysize_px)

        xy = np.meshgrid(x.T, y.T) @ self.rotation_matrix
        xy += self.pos_nm
        return xy

    def abs2topo(self, abspos):
        return (abspos - self.pos_nm) @ self.rotation_matrix.T + self.size_nm/2

    def nm2px(self, value):
        return np.mean(self.size_px / self.size_nm)



