""" Open and parse.dat file. """

import os.path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .helper import get_logger
from .helper.dataset import DataSetBase

log = get_logger(__name__)
from .sxmfile import SxmFile


class SxmDataSet(DataSetBase):
    def __init__(self, *things):

        self.series = dict()
        self._objs = set()
        super().__init__(
            *things,
            cls=SxmFile,
            opener=SxmFile,
            ext='.sxm',
            sort_key='datetime',
            index_key='uid'
        )

    # def get_data(self):
    #    obj = list(self._objs)

    #    data = pd.DataFrame({
    #        'path': [bsp.filename for bsp in specs],
    #        'obj': [bsp for bsp in specs],
    #        'start_time': [bsp.start_time for bsp in specs],
    #        'end_time': [bsp.end_time for bsp in specs],
    #        'V_start': [bsp.v_start for bsp in specs],
    #        'V_end': [bsp.v_end for bsp in specs],
    #        'pixels': [bsp.pixels for bsp in specs],
    #        'name': [bsp.name for bsp in specs],
    #        'serie_name': [bsp.serie_name for bsp in specs],
    #        'serie_number': [bsp.serie_number for bsp in specs],
    #    })
    #    data.sort_values('start_time', inplace=True)
    #    data.set_index('name', inplace=True)
    #
    #    return

    def generate_images(self, output_directory='sxm-images', figsize=None):
        N = len(self)
        s = ''.join(['Generating image for %s (%',
                     '%d' % (np.log10(N) + 1),
                     'd/%d)'])
        os.makedirs(output_directory, exist_ok=True)
        if figsize is None:
            figsize = (10, 10)

        fig = Figure(figsize=figsize)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        for i, obj in enumerate(self.objs):
            log(s, obj.uid, i + 1, N)
            fn = os.path.join(output_directory, obj.uid + '.png')

            obj.plot(ax=ax)
            fig.savefig(fn, bbox_inches='tight', dpi=150)
            ax.clear()
