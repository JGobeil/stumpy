import os

import matplotlib.pyplot as plt
import numpy as np

from .plotting import get_figsize
from .sxmfile import SxmFile
from .datfile import BiasSpec

default_sxmplot_config = {
    'info': 'minimal',
    'size': 'medium',
    'cmap': 'hsv',
}


class Opener:
    def __init__(self,
                 datasrc,
                 sxmname,
                 sxmplot_config=None,
                 specsname=None
                 ):

        self.datasrc = datasrc
        self.sxmname = sxmname
        self.config_sxmplot = default_sxmplot_config.copy()
        if sxmplot_config is not None:
            self.config_sxmplot.update(sxmplot_config)

        self.specsname = specsname

    def plot(self, *numbers, nrows=1, **kwargs):
        """Backward compatibility"""
        return self.plot_sxm(*numbers, nrows=nrows, **kwargs)

    def plot_sxm(self, *numbers, nrows=1, **kwargs):
        sxms = self.getsxm(*numbers)

        cfg = self.config_sxmplot.copy()
        cfg.update(**kwargs)

        N = len(numbers)
        if N == 1:
            return sxms.plot(**cfg)
        else:
            ncols = int(np.ceil(N / nrows))
            figsize = get_figsize(cfg['size'])

            fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            axis = axis.flatten()

            for i, ax in enumerate(axis):
                if i < N:
                    sxms[i].plot(ax=ax, **cfg)
                else:
                    ax.axis('off')
            return fig, axis

    def _getfn_sxm(self, nb):
        return os.path.join(self.datasrc, self.sxmname + "%.3d" % nb + '.sxm')

    def _getfn_specs(self, nb):
        return os.path.join(self.datasrc, self.specsname + "%.3d" % nb + '.dat')

    def getsxm(self, *numbers):
        if self.sxmname is None:
            print("Error: Need to specify a sxmname")
            return

        sxms = [SxmFile(self._getfn_sxm(nb)) for nb in numbers]

        N = len(numbers)
        if N == 1:
            return sxms[0]
        else:
            return sxms

    def getspecs(self, *numbers):
        if self.specsname is None:
            print("Error: Need to specify a specsname")
            return

        specs = [BiasSpec(self._getfn_specs(nb)) for nb in numbers]

        N = len(numbers)
        if N == 1:
            return specs[0]
        else:
            return specs

