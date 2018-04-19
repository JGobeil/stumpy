""" Topo scan viewing and basic information. Modifying method should be in
seperated files and use the decorator @topomod to link it to the class Topo."""

import matplotlib.cm
import numpy as np
from collections import ChainMap
import base64
import io
import os

from ..helper.lazy import lazy_property
from ..plotting import create_figure
from ..plotting import no_axis, no_grid, no_ticks
from ..plotting import add_title
from ..helper import get_logger
log = get_logger(__name__)


from .. import topo_info_formatter

class Topo:
    """ A class with the Topo 2D data, it size and pos. ."""

    def __init__(self, src):
        """ Initialize a topo. """
        self.src = src
        self.sxm = src.sxm
        self.channel = src.channel
        self.size_nm = src.sxm.size_nm
        self.pos_nm = src.sxm.pos_nm
        self.angle = src.sxm.angle
        self.plot_defaults = self.src.plot_defaults.new_child()

    def set_plot_defaults(self, **kwargs):
        self.plot_defaults.update(**kwargs)

    @property
    def data(self):
        """ The raw data as numpy.array"""
        return self.src.data

    @property
    def size_px(self):
        """ The size in pixels as numpy.array"""
        return np.array(self.data.shape)

    @property
    def size_nm_str(self):
        nmx, nmy = self.size_nm
        return "%.5g nm" % nmx if nmx == nmy else "%.5gx%.5g nm" % (nmx, nmy)

    @property
    def size_px_str(self):
        pxx, pxy = self.size_px
        return "%d px" % pxx if pxx == pxy else "%dx%d px" % (pxx, pxy)

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

    @property
    def modstr(self):
        mod = []
        src = self
        while isinstance(src, Topo):
            mod.append(str(src))
            src = src.src
        return '\n'.join(mod)

    def __str__(self):
        return "Channel{name: %s}" % self.channel.name

    def plot(self, ax=None, **kwargs):

        cha = self.channel
        sxm = self.sxm

        params = dict(**self.plot_defaults)
        params.update(**kwargs)

        info = params.pop('info')
        show_axis = params.pop('show_axis')
        boxed = params.pop('boxed')
        size = params.pop('size')
        pyplot = params.pop('pyplot')
        dpi = params.pop('dpi')
        save = params.pop('save')
        tight = params.pop('tight')
        savepath = params.pop('savepath')

        if ax is None:
            figure = create_figure(size=size, pyplot=pyplot, dpi=dpi,
                                   shape='square')
            ax = figure.add_subplot(111)

        x1, y1 = (0, 0)
        x2, y2 = np.array([x1, y1]) + self.size_nm

        ax.imshow(
            self.data,
            extent=(x1, x2, y1, y2),
            **params
        )


        if info is None:
            fmtstr = ""
        elif info in topo_info_formatter:
            fmtstr = topo_info_formatter[info]
        else:
            fmtstr = info

        infostr = fmtstr.format(
            sxm=self.sxm,
            channel=self.channel,
            topo=self
        )

        add_title(ax, infostr)

        if not show_axis:
            no_axis(ax)
        if boxed:
            no_ticks(ax)
        no_grid(ax)

        if tight:
            ax.get_figure().tight_layout()

        if save is not False:
            if save is True:
                fn = "{sxm.filename}_{channel.name}.png".format(
                    sxm=self.sxm,
                    channel=self.channel,
                    topo=self
                )
            else:
                fn = save

            if savepath:
                os.makedirs(savepath, exist_ok=True)
                fn = os.path.join(savepath, fn)

            ax.get_figure().savefig(fn)
            log.info("Plot saved on %s", fn)
        return ax

    def _repr_html_(self):
        return ''.join([
            '<img src="data:image/png;base64,',
            self.get_base64_plot(),
            '" />'])

    def get_base64_plot(self, **kwargs):
        ax = self.plot(pyplot=False, **kwargs)
        bts = io.BytesIO()
        ax.get_figure().savefig(bts, format='png')
        bts.seek(0)
        return base64.b64encode(bts.getvalue()).decode()


