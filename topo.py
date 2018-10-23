""" Topo scan viewing and basic information. Modifying method should be in
seperated files and use the decorator @topomod to link it to the class Topo."""

import matplotlib.cm as mplcm
import numpy as np
from collections import ChainMap
import base64
import io
import os
from collections import UserList
import fnmatch
from scipy.interpolate import RectBivariateSpline

from helper.lazy import lazy_property
from plotting import create_figure
from plotting import no_axis, no_grid, no_ticks
from plotting import add_title
from plotting import get_figsize
from helper import get_logger
from helper.image import writetopo
from helper.image import encodetopo
from helper.image import normdata
log = get_logger(__name__)

from . import topoinfo_formatter


class Topo:
    def __new__(cls, src, *args, **kwargs):
        """ If source is an iterable create an iterable of Topo. Allow batch
        processing. """
        try:
            obj = src.__class__([cls(s, *args, **kwargs) for s in src])
        except TypeError:
            obj = super().__new__(cls)
        return obj

    """ A class with the Topo 2D data, it size and pos. ."""
    def __init__(self, src, *args, **kwargs):
        """ Initialize a topo. """
        self.src = src
        self.sxm = src.sxm
        self.channel = src.channel
        self.pos_nm = src.pos_nm.copy()
        self.size_nm = src.size_nm.copy()
        self.angle = src.angle
        self.name = src.name
        self.minmax = None
        self.plot_defaults = self.src.plot_defaults.new_child()

    def set_plot_defaults(self, **kwargs):
        self.plot_defaults.update(**kwargs)

    @property
    def bl_corner(self):
        return self.pos_nm - self.size_nm/2 @ self.inv_rotation_matrix

    @property
    def tr_corner(self):
        return self.pos_nm + self.size_nm/2 @ self.inv_rotation_matrix

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
    def inv_rotation_matrix(self):
        """ Rotation matrix using the angle of the scan."""
        theta = -np.radians(self.angle)
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

    def map_topo2plot(self, pos):
        "Conversion from topo space to plot space"
        p = np.asarray(pos)
        return (p - self.pos_nm) @ (self.rotation_matrix) + self.size_nm/2

    def map_plot2topo(self, pos):
        "Conversion from plot space to topo space"
        p = np.asarray(pos)
        return (p - self.size_nm/2) @ self.inv_rotation_matrix + self.pos_nm

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


    @property
    def imdata(self):
        return np.flipud(self.data.T)

    def plot(self, ax=None, **kwargs):
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
        fontdict_title = params.pop('fontdict_title')
        absolute_pos = params.pop('absolute_pos')

        if ax is None:
            figure = create_figure(size=size, pyplot=pyplot, dpi=dpi,
                                   shape='square')
            ax = figure.add_subplot(111)

        if absolute_pos:
            x1, y1 = self.pos_nm - self.size_nm/2
            x2, y2 = self.pos_nm + self.size_nm/2
        else:
            x1, y1 = (0, 0)
            x2, y2 = self.size_nm

        mm = self.minmax
        ax.imshow(
            self.imdata,
            extent=(x1, x2, y1, y2),
            vmin=None if mm is None else mm[0],
            vmax=None if mm is None else mm[1],
            **params
        )


        if info is None:
            fmtstr = ""
        elif info in topoinfo_formatter:
            fmtstr = topoinfo_formatter[info]
        else:
            fmtstr = info

        infostr = fmtstr.format(
            sxm=self.sxm,
            channel=self.channel,
            topo=self
        )

        ax.set_title(infostr, fontdict=fontdict_title)

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

    def __repr__(self):
        return self.name

    def get_base64_plot(self, **kwargs):
        ax = self.plot(pyplot=False, **kwargs)
        bts = io.BytesIO()
        ax.get_figure().savefig(bts, format='png')
        bts.seek(0)
        return base64.b64encode(bts.getvalue()).decode()

    def get_base64_image(self, cmap=None):
        if cmap is None:
            cmap = self.plot_defaults['cmap']
        bts = encodetopo(data=self.imdata, cmap=cmap,
                         minmax=self.minmax)
        return base64.b64encode(bts).decode()

    def write(self, filename, cmap=None):
        if cmap is None:
            cmap = self.plot_defaults['cmap']
        writetopo(self.imdata, filename, cmap=cmap,
                  minmax=self.minmax)

    def is_in(self, xy):
        p = self.map_topo2plot(xy)
        return np.all(p > 0) and np.all(p < self.size_nm)

class InterpolatedTopo(Topo):
    def __init__(self, src):
        super().__init__(src)
        self.pixels = src.size_px.max()

        self._interpolation = RectBivariateSpline(
            np.linspace(0, src.size_nm[0], src.size_px[0]),
            np.linspace(0, src.size_nm[1], src.size_px[1]),
            src.data,
        )
        self.cuts = (0, src.size_nm[0], 0, src.size_nm[1])

    @property
    def data(self):
        xmin, xmax, ymin, ymax = self.cuts
        return self._interpolation(
            np.linspace(xmin, xmax, self.pixels[0]),
            np.linspace(ymin, ymax, self.pixels[1]),
        )

    @property
    def size_nm(self):
        return self._size_nm

    @size_nm.setter
    def size_nm(self, s):
        #poff = self.pos_offset
        #c0, c2 = (self.src.size_nm - s) / 2 + poff
        #c1, c3 = (self.src.size_nm + s) / 2 + poff
        #self.cuts = (c0, c1, c2, c3)
        self._size_nm = np.asarray(s)

    @property
    def pos_offset(self):
        return self.pos_nm - self.src.pos_nm

    @pos_offset.setter
    def pos_offset(self, poff):
        self.pos_nm = self.src.pos_nm + poff

    @property
    def cuts(self):
        x0, y0 = self.src.map_topo2plot(self.bl_corner)
        x1, y1 = self.src.map_topo2plot(self.tr_corner)

        return (x0, x1, y0, y1)

    @cuts.setter
    def cuts(self, cuts):
        x0, x1, y0, y1 = cuts
        self.size_nm = np.asarray((x1 - x0, y1 - y0))

        self.pos_nm = (self.src.map_plot2topo((x0, y0)) +
                       self.src.map_plot2topo((x1, y1))) / 2

    @property
    def pixels(self):
        ratio = np.asarray(self.size_nm / self.size_nm.max())
        return (self._pixels * ratio).astype(int)

    @pixels.setter
    def pixels(self, N):
        self._pixels = np.max(N)


class TopoSet(UserList):
    def __init__(self, topos=[]):
        toadd = []
        for t in topos:
            try:
                toadd.extend(t)
            except TypeError:
                toadd.append(t)
        super().__init__(toadd)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        names = self.names
        topodict = self.topo_dict
        if item in names:
            return topodict[item]
        match = fnmatch.filter(names, '*'+item+'*')
        if len(match) < 1:
            raise IndexError()
        if len(match) > 1:
            return TopoSet([topodict[m] for m in match])
        else:
            return topodict[match[0]]


    def filter_by_name(self, filter):
        if isinstance(filter, str):
            filter = [filter, ]

        names = self.names
        topodict = self.topo_dict

        filtered = [fnmatch.filter(names, '*'+f+'*') for f in filter]
        flatten = [f for sublist in filtered for f in sublist]
        names_set = set(flatten)
        return TopoSet([topodict[n] for n in names if n in names_set])

    def plot(self, ncols=3, **kwargs):
        params = kwargs.copy()

        N = len(self)
        nrows = int(np.ceil(N/ncols))

        if 'size' not in params:
            params['size'] = 'small'

        sc, sr = get_figsize(size=params['size'], shape='square')
        params['size'] = (sc*ncols, sr*nrows)

        fig = create_figure(**params)
        axes = fig.subplots(nrows=nrows, ncols=ncols)

        for i, ax in enumerate(axes.flatten()):
            if i < N:
                self[i].plot(ax=ax, **params)
            else:
                no_axis(ax)
        return fig, axes

    def __add__(self, other):
        return TopoSet(self.data + other.data)

    def __repr__(self):
        return "ChannelSet(%s)" % ', '.join([c.name for c in self])

    @property
    def names(self):
        return [t.name for t in self.data]

    @property
    def topo_dict(self):
        d = {t.name: t for t in self.data}
        d.update({i: t for i, t in enumerate(self.data)})
        return d

    def _repr_html_(self):
        return ''.join([
            '<img src="data:image/png;base64,',
            self.get_base64_plot(),
            '" />'])

    def get_base64_plot(self, **kwargs):
        fig, ax = self.plot(pyplot=False, **kwargs)
        bts = io.BytesIO()
        fig.savefig(bts, format='png')
        bts.seek(0)
        return base64.b64encode(bts.getvalue()).decode()
