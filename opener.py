import os

import matplotlib.pyplot as plt
import numpy as np

from .plotting import get_figsize
from .sxmfile import SxmFile

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
                 ):

        self.datasrc = datasrc
        self.sxmname = sxmname
        self.config_sxmplot = default_sxmplot_config.copy()
        if sxmplot_config is not None:
            self.config_sxmplot.update(sxmplot_config)

    def plot(self, *numbers, nrows=1, **kwargs):
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

    def _getfn(self, nb):
        return os.path.join(self.datasrc, self.sxmname + "%.3d" % nb + '.sxm')

    def getsxm(self, *numbers):
        sxms = [SxmFile(self._getfn(nb)) for nb in numbers]

        N = len(numbers)
        if N == 1:
            return sxms[0]
        else:
            return sxms
