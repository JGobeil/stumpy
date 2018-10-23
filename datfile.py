"""Open and parse .dat file"""

import os.path
from collections import Counter
from types import SimpleNamespace
import io
import base64

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.style import context as mpl_context

from .helper import get_logger
from .helper import lazy_property
from .helper.fileparser import TabHeaderFile
from .helper.fileparser import Parse

from .plotting import get_figsize
from .plotting import create_figure

log = get_logger(__name__)

class GenericDatFile(TabHeaderFile):
    header_end = '[DATA]'
    dataoffset = 0

    @lazy_property
    def xyz_nm(self):
        return np.array((self.x_nm, self.y_nm, self.z_nm))

    # alias
    @lazy_property
    def pos_nm(self):
        return self.xyz_nm

    @lazy_property
    def x_nm(self):
        return float(self.header['X (m)'])*1e9

    @lazy_property
    def y_nm(self):
        return float(self.header['Y (m)'])*1e9

    @lazy_property
    def xy_nm(self):
        return np.array((self.x_nm, self.y_nm))

    @lazy_property
    def z_nm(self):
        return float(self.header['Z (m)'])*1e9

    @lazy_property
    def datetime(self):
        return Parse.datetime(self.header['Date'])

    @lazy_property
    def number(self):
        return int(self.filename[-7:-4])


class BiasSpec(GenericDatFile):

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
        'dIdmV': 'dI/dV (nA/mV)',  # 'normalized' dI/dV
        'dIdV_LI': 'dI/dV (nA/V) (from LI)',
        'dIdmV_LI': 'dI/dV (nA/mV) (from LI)',
        'dI_LI_ratio': 'dI_vs_LI_ratio',  # 'ratio between numeric and Lock-In
        'mV': 'Bias (mV)',  # 'ratio between numeric and Lock-In
    }

    def __init__(self,
                 filename,
                 LI='LIY',
                 noise_limits=(0.25, 0.75),
                 LI_sens=50e-3,
                 LI_amplitude=0.2,
                 divider=100,
                 constant_current=False,
                 ):
        # read file and parse header
        super().__init__(filename)

        # use an attribute so it can be changed on specific file if needed
        self._cnsl = BiasSpec.channel_names_search_list
        self._cfn = BiasSpec.calculated_field_names

        # use to cut the current data to limit the noise at high current
        self.noise_limits = np.array(noise_limits)

        self.LI_sensitivity = LI_sens
        self.divider = divider
        self.LI_amplitude = LI_amplitude
        self.constant_current = constant_current

        if LI not in ['LIY', 'LIX']:
            self.is_ok = False
            #log.wrn("Lock-In channes should be 'LIX' or 'LIY'")

        if 'Bias Spectroscopy>Channels' not in self.header:
            # Not a Bias Spec file
            self.is_ok = False
            #log.wrn("%s is not as BiasSpec file.", self.filename)
            return

        self.keys = self.infer_keys(LI=LI)
        log.dbg("Opened: '%s' as BiasSpec (LI: %s; Bias: %s)",
                self.filename, self.keys.LI, self.keys.V)

    def plot(self, title=None, ax=None, save=False, size=None,
             x=None, y=None, xlabel=None, ylabel=None,
             pyplot=True, dpi=100,
             **kwargs,):
        if 'figsize' not in kwargs and ax is None:
            kwargs['figsize'] = get_figsize(size)

        if x is None:
            x = self.keys.mV
        if y is None:
            y = self.keys.dIdmV
        if xlabel is None:
            xlabel = 'Bias [mV]'
        if ylabel is None:
            ylabel = 'dI/dV [nA/mV]'

        if ax is None:
            figure = create_figure(size=size, pyplot=pyplot, dpi=dpi,
                                   shape='golden')
            ax = figure.add_subplot(111)

        ax = self.data.plot(x=x, y=y, ax=ax, **kwargs)

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

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
        dV = (df[k.V].max() - df[k.V].min()) / (N-1)

        # limits noise issues

        ml = np.round(self.noise_limits * N).astype(int)


        df[k.mV] = df[k.V]*1000
        df[k.NdIdV] = np.gradient(1e9*df[k.I], dV, edge_order=2)
        
        self.ratio_lock_in = self.LI_sensitivity / (2*self.LI_amplitude)
        self.ratio_numeric = abs((df[k.NdIdV] / df[k.LI])[ml[0]:ml[1]].mean())
        
        df[k.dIdV] = df[k.LI] * self.ratio_numeric
        df[k.dIdmV] = df[k.dIdV] / 1000
        
        df[k.dIdV_LI] = df[k.LI] * self.ratio_lock_in 
        df[k.dIdmV_LI] = df[k.dIdV_LI] / 1000
        
        return df
    
    @property
    def ratio_info_str(self):
        self.data
        return "%g / %g (num / li) ;  %g / %g (li / est)" % (
        self.ratio_numeric, self.ratio_lock_in, 
        self.LI_sensitivity, self.LI_sensitivity_estimation)
    
    @property
    def LI_sensitivity_estimation(self):
        self.data
        return self.ratio_numeric*2*self.LI_amplitude 

    def iplot(self):
        import ipywidgets as w
        from IPython.display import display

        xselect = w.Select(
            description='x',
            options=self.keys.__dict__.values(),
            value=self.keys.mV,
        )
        yselect = w.Select(
            description='y',
            options=self.keys.__dict__.values(),
            value=self.keys.dIdmV,
        )

        html = w.HTML()

        def change(*args):
            html.value = "".join([
            "<img style='display:block;width:600px;'",  # width:100px;height:100px;'
            "src='data:image/png;base64, ",
            self.get_base64_plot(
                x=xselect.value,
                y=yselect.value,
            ),
            "' />",
            ])

        xselect.observe(change, 'value')
        yselect.observe(change, 'value')

        change()
        display(w.VBox([xselect, yselect, html]))


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

    def get_base64_plot(self, **kwargs):
        ax = self.plot(pyplot=False, **kwargs)
        bts = io.BytesIO()
        ax.get_figure().savefig(bts, format='png')
        bts.seek(0)
        return base64.b64encode(bts.getvalue()).decode()

