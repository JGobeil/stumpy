""" Topo scan viewing and basic information. Modifying method should be in
seperated files and use the decorator @topomod to link it to the class Topo."""

import matplotlib.cm
import numpy as np

from ..helper.lazy import lazy_property

default_cmap = matplotlib.cm.get_cmap('Blues_r')

from functools import wraps
from inspect import getcallargs


def topomod(topo_method=True):
    """ Decorator of Topo modifcation function. This help
    keep track of what have happened to the Topo and automatically
    add the function to the Topo class when imported. A dictionary linking
    the function name with the function is saved as a Topo class attribute."""

    def deco(modfunc):
        @wraps(modfunc)
        def func(*args, **kwargs):
            # call the function
            out = modfunc(*args, **kwargs)

            # save info about the function
            out.modfunc = modfunc.__name__
            out.modfunc_args = getcallargs(modfunc, *args, **kwargs)

            return out

        func.arg0_is_topo = topo_method
        if topo_method:
            setattr(Topo, modfunc.__name__, func)

        # save function in the db
        Topo.mod_db[modfunc.__name__] = func
        return func

    return deco


class Topo:
    """ A class with the Topo 2D data, it size and pos. Topo.mod_db is a
    dictionary with the Topo modifying function."""

    mod_db = {}

    def __init__(self, data, size_nm, pos_nm):
        """ Create a topo object.

        Parameters
        ----------
        data: 2D list or array
            2D data of the surface in nm.
        size_nm: tuple-like
            x and y size in nm
        pos_nm: tuple-like
            x and y positon in nm
        """
        self.data = data
        self.size_nm = size_nm
        self.pos_nm = pos_nm

    @property
    def size_nm(self):
        """ The size in nanometers as numpy.array"""
        return self._size_nm

    @size_nm.setter
    def size_nm(self, size_nm):
        self._size_nm = np.array(size_nm)

    @property
    def pos_nm(self):
        """ The position in nanometers as numpy.array"""
        return self._pos_nm

    @pos_nm.setter
    def pos_nm(self, pos_nm):
        self._pos_nm = np.array(pos_nm)

    @property
    def data(self):
        """ The raw data as numpy.array"""
        return self._data

    @data.setter
    def data(self, data):
        self._data = np.array(data)

    @property
    def size_px(self):
        """ The size in pixels as numpy.array"""
        return np.array(self.data.shape)

    def realpos_to_pxpos(self, realpos):
        """ Conversion from real position (nm) to pixels positions(float)"""
        return realpos * self.conv_real_to_px

    def realpos_to_px(self, realpos):
        """ Conversion from real position (nm) to pixels(int)"""
        return np.rint(realpos * self.conv_real_to_px).astype(int)

    def pxpos_to_realpos(self, pxpos):
        """ Conversion from pixels positions to real position (nm)"""
        return pxpos / self.conv_real_to_px

    @lazy_property
    def conv_real_to_px(self):
        """ Conversion real to pixels constants"""
        return (self.size_px - 1) / self.size_nm

    def get_realpos_of_all_pixels(self):
        """ The real position (nm) of all pixels."""
        return self.pxpos_to_realpos(
            np.array(
                np.meshgrid(
                    np.arange(self.size_px[0]),
                    np.arange(self.size_px[1])
                )
            ).T
        )

    @lazy_property
    def realpos_of_all_pixels(self):
        """ The real position (nm) of all pixels as a lazy property. """
        return self.get_realpos_of_all_pixels()
