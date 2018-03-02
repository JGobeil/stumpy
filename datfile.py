"""Open and parse.dat file"""

import os.path
from collections import Counter
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.style import context as mpl_context

from .helper import get_logger
from .helper import lazy_property
from .helper.dataset import DataSetBase
# from .helper.fileparser import Parse
from .helper.fileparser import TabHeaderFile

log = get_logger(__name__)


class DatDataSet(DataSetBase):
    def __init__(self, *things, analyse=True, timesplit=False):

        super().__init__(
            *things,
            cls=BiasSpec,
            opener=open_datfile,
            ext='.dat',
            sort_key='start_time',
            index_key='name',
        )

        self.timesplit = timesplit

        self.series = dict()
        if analyse:
            self.analyse()

    def analyse(self):
        if len(self) == 0:
            # empty DatDataSet
            print("EMPTY")
            return

        self.series = dict()
        single = []

        uniques = self.data[[
            'serie_name',
            'V_start',
            'V_end',
            'pixels'
        ]].copy()
        uniques['name'] = [str(row.values) for i, row
                           in uniques.iterrows()]
        count = Counter(uniques.name)
        serie_count = Counter()

        if np.all(np.array(list(count.values())) == 1):
            # single values DatDataSet
            print("SINGLE VALUE")
            return

        if len(count) == 1:
            # single serie DatDataSet
            log.dbg("Single name DatDataSet (%s)", list(count.keys())[0])
            if not self.timesplit:
                return

            start_time = np.roll(self.data.start_time.copy(), -1)
            end_time = self.data.end_time.copy()
            start_time[-1] = start_time[-2]
            end_time[-1] = end_time[-2]
            timedelta = (start_time - end_time) / np.timedelta64(1, 's')

            ts = timedelta
            # little trick to find peak
            ts = np.abs(ts - np.mean(ts))
            ts = np.abs(ts - np.mean(ts))
            ts = np.abs(ts - np.mean(ts))
            splits = np.argwhere(ts > 3 * np.mean(ts))

            self.timesplitted = DatDataSet(analyse=False)
            self.timesplitted.timedelta = timedelta
            self.timesplitted.splits = splits

        for name, nb in count.items():
            data = self.data[uniques.name == name]
            if nb == 1:
                single.append(*data.obj)
            else:

                sname = data.iloc[0].serie_name
                serie_count.update([sname])
                i = serie_count[sname]

                self.series["%s_%.3i" % (sname, i)] = DatDataSet(*data.obj)

        self.single = DatDataSet(*single, analyse=False)

        log("Found %d data files [%i single specs - %i series]",
            len(self), len(self.single), len(self.series))

        # remove _001 from serie when only one
        # for sname, c in serie_count.items():
        #    if c == 1:
        #        self.series[sname] = self.series["%s_001" % sname]
        #        del self.series["%s_001" % sname]

    @property
    def paths(self):
        return self.data['path']

    @property
    def by_path(self):
        return {path: obj for path, obj in zip(self.paths, self.objs)}

    def merge(self, other, analyse=True):
        super().merge(other)
        if analyse:
            self.analyse()

    def plot2D(self):
        objs = self.objs

        data = np.array([obj.data[obj.keys.dIdV] for obj in objs])
        ax = plt.imshow(data)

        return ax

    def plot_timesplits(self):
        try:
            ts = self.timesplitted
        except AttributeError:
            log.err("No time split.")
            return

        timedelta = ts.timedelta
        splits = ts.splits

        for s in splits:
            plt.axvline(s)

        return plt.plot(timedelta)


def open_datfile(filename, common_path=None):
    """ Open a .dat file. Return an object corresponding to the appropriate
    experiment. Implemented for 'bias spectroscopy'.

    Parameters
    ----------
    filename: str
        file to open

    Returns
    -------
    BiasSpec or ...?

    """
    if common_path is None:
        fn = filename
    else:
        fn = os.path.join(common_path, filename)

    with open(fn) as f:
        exp = f.readline().strip().split('\t')[-1].strip()

    if exp == 'bias spectroscopy':
        return BiasSpec(filename, common_path)
    else:
        return GenericDatFile(filename, common_path)


class BiasSpec(TabHeaderFile):
    header_end = '[DATA]'
    dataoffset = 0

    # columns name to search for in file. Go through a list until a match.
    channel_names_search_list = {
        'I': ['Current (A)', ],
        'V': ['Bias calc (V)', 'Bias (V)', 'Bias w LI (V)', ],
        'LIX': ['Lock-In X (V)',
                'Lock-in_X (V)',
                'LI Demod 1 X (A)', ],
        'LIY': ['Lock-In Y (V)',
                'Lock-in_Y (V)',
                'LI Demod 1 Y (A)', ],
    }

    # column names for the calculated data
    calculated_field_names = {
        'NdIdV': 'NdI/dV (V)',  # numerical dI/dV (from current)
        'dIdV': 'dI/dV (nA/V)',  # 'normalized' dI/dV
        'dI_LI_ratio': 'dI_vs_LI_ratio',  # 'ratio between numeric and Lock-In
    }

    def __init__(self,
                 filename, common_path=None,
                 LI='LIY',
                 noise_limits=(0.25, 0.75),
                 ):
        super().__init__(filename, common_path)

        # use an attribute so it can be changed on specific file if needed
        self._cnsl = BiasSpec.channel_names_search_list
        self._cfn = BiasSpec.calculated_field_names

        # use to cut the current data to limit the noise at high current
        self._noise_limits = np.array([0.25, 0.75])

        if LI not in ['LIY', 'LIX']:
            log.wrn("Lock-In channes should be 'LIX' or 'LIY'")
        self.keys = self.infer_keys(LI=LI)

        log.dbg("Opened: '%s' as BiasSpec (LI: %s; Bias: %s)",
                self.filename, self.keys.LI, self.keys.V)

    def plot(self, title=None, ax=None, save=False, **kwargs):
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (10, 7)

        with mpl_context('ggplot'):
            ax = self.dIdV.plot(
                x=self.keys.V, y=self.serie_number, ax=ax, **kwargs
            )
            if title is not None:
                ax.get_figure().set_title(title)
            ax.set_xlabel('Bias [V]')
            ax.set_ylabel('dI/dV [nA/V]')

            if save is not False:
                if save is True:
                    if title is not None:
                        filename = title + '.png'
                    else:
                        filename = self.name + '.png'
                else:
                    filename = save
                ax.get_figure().savefig(filename)

            return ax

    @property
    def dIdV(self):
        return self.data.rename(columns={self.keys.dIdV: self.serie_number})

    @property
    def name(self):
        return self.serie_name + "%.3i" % self.serie_number

    def infer_keys(self, LI='LIY'):
        """ Infer the key names from the file header."""
        k = {}
        for key, names in self._cnsl.items():
            for name in names:
                if name in self.channel_names:
                    k[key] = name
                    break
        k.update(**self._cfn)
        if LI in k:
            k['LI'] = k[LI]

        # Check
        for field, name in [
            ('V', 'Bias'),
            ('I', 'Current'),
        ]:
            if field not in k:
                log.wrn("%s: No '%s' field found.\n"
                        '\tsearched for :%s\n'
                        '\tfield in file: %s',
                        self.filename, name,
                        self._cnsl[field], self.channel_names)
        if 'LI' not in k:
            log.wrn("%s: No 'Lock-In' field.", self.filename)
            k['LI'] = None
        return SimpleNamespace(**k)

    @lazy_property
    def channel_names(self):
        """ List of channels names that can be found in file."""
        header_names = [s.strip() for s in
                        self.header['Bias Spectroscopy>Channels'].split(';')]

        # 'Bias calc (V)' is in file but not in the header.
        return ['Bias calc (V)', ] + header_names

    def get_data(self):
        k = self.keys

        df = super().get_data().sort_values(k.V)

        N = len(df[k.V])
        dV = (df[k.V].max() - df[k.V].min()) / N

        # limits noise issues

        ml = np.round(self._noise_limits * N).astype(int)

        df[k.NdIdV] = np.gradient(df[k.I], dV, edge_order=2)
        df[k.dI_LI_ratio] = df[k.NdIdV] / df[k.LI]
        df.ratio = df[k.dI_LI_ratio][ml[0]:ml[1]].mean()
        df[k.dIdV] = df[k.LI] * df.ratio * 1e9

        return df

    def __repr__(self):
        return "%s (%gV .. %gV)" % (
                self.serie_number, self.v_start, self.v_end)

    @lazy_property
    def calibration(self):
        return float(self.header['Bias>Calibration (V/V)'])

    @lazy_property
    def v_start(self):
        return float(self.header['Bias Spectroscopy>Sweep Start (V)'])

    @lazy_property
    def v_end(self):
        return float(self.header['Bias Spectroscopy>Sweep End (V)'])

    @lazy_property
    def serie_name(self):
        return os.path.basename(self.filename)[:-7]

    @lazy_property
    def serie_number(self):
        return int(os.path.basename(self.filename)[-7:-4])

    @lazy_property
    def pixels(self):
        return int(self.header['Bias Spectroscopy>Num Pixel'])


class GenericDatFile(TabHeaderFile):
    header_end = '[DATA]'
    dataoffset = 2
