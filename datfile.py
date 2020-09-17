"""Open and parse .dat file"""

import os.path
from collections import Counter
from types import SimpleNamespace
import io
import base64
from collections import ChainMap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.style import context as mpl_context
from lazy import lazy

from .helper import get_logger
from .helper import lazy_property
from .helper.fileparser import TabHeaderFile
from .helper.fileparser import Parse
from .helper.units import unit_factor, unit_names, unit_symbol

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
        if 'Date' in self.header:
            return Parse.datetime(self.header['Date'])
        if 'Saved Date' in self.header:
            return Parse.datetime(self.header['Saved Date'])
        return None

    @lazy_property
    def number(self):
        return int(self.filename[-7:-4])
    
    @lazy_property
    def experiment(self):
        return self.header["Experiment"]
    
    @lazy_property
    def info(self):
        return {
            "x (nm)": self.x_nm,
            "y (nm)": self.y_nm,
            "datetime": self.datetime,
            "experiment": self.experiment
        }
    
    def __repr__(self):
        return "GenericDatFile(%s)" % self.filename

class BiasSpec(GenericDatFile):

    # columns name to search for in file. Go through a list until a match.
    search_names = ChainMap({
        'Current': ['Current (A)', ],
        'Bias': ['Bias w LI (V)', 'Bias calc (V)', 'Bias (V)', ],
        'LIX': ['Lock-In X (V)',
                'Lock-in_X (V)',
                'LI Demod 1 X (A)', ],
        'LIY': ['Lock-In Y (V)',
                'Lock-in_Y (V)',
                'LI Demod 1 Y (A)', ],
        'LIR': ['Lock-In R (V)',
                'Lock-in_R (V)',
                'LI Demod 1 R (A)', ],
        'Z': ['Z (m)', ],
        'Field': ['B Field (T)', ],
    })

    # column names for the calculated data
    field_names = ChainMap({
        'Current': 'Current [nA]',
        'Bias': 'Bias [mV]',
        'Z': 'Z [nm]',
        'LIX': 'Lock-In X [V]',
        'LIY': 'Lock-In Y [V]',
        'LIR': 'Lock-In R [V]',
        'NdIdV': 'NdI/dV [S]',  # numerical dI/dV (from current)
        'dIdV': 'dI/dV [μS]',
        'd2IdV2': '$d^2I/dV^2 (\muS)$',
        'dIdV_LI': 'dI/dV [μS] (from lock-in)',
        'dIdV_FIT': 'dI/dV [μS] (from numerical fit)',
        'Field': 'Field [mT]',
        'Gauss': "dI/dV [μS] (gaussian)"
    })

    def __init__(self,
                 filename,  # the field to open
                 LI='LIY',  # the data channel to use (see BiasSpec.search_names)
                 default_channel='dIdV',  # default channel to plot
                 force_numerical_dIdV=False,  # use numerical derivation even if lock in info is availible
                 noise_limits=(0.25, 0.75),  # limit to consider when fitting the numerical derivation
                 LI_sensivity=None,  # sensitivity of the lock in
                 LI_amplitude=None,  # amplitude of the modulation of the lock in
                 divider=100,  # value of the current divider
                 constant_current=False,  # if the spectroscopy is at constant current
                 search_names=None,  # override some search names
                 field_names=None,  # override some field names
                 correct_bias_offset=False,  # automatic bias offset correction (True=automatic, float=manual)
                 hamming=0, # hamming filtering of the dI/dV
                 legend=True,  # False for no legend
                 ):
        # read file and parse header
        super().__init__(filename)


        self.LI = LI
        self.default_channel = default_channel
        self.force_numerical_dIdV = force_numerical_dIdV
        self.noise_limits = np.asarray(noise_limits)
        self.LI_sens = LI_sensivity
        self.LI_ampl = LI_amplitude
        self.divider = divider
        self.constant_current = constant_current
        self.search_names = BiasSpec.search_names.new_child(search_names)
        self.field_names = BiasSpec.field_names.new_child(field_names)
        self.correct_bias_offset = correct_bias_offset
        self._hamming = hamming
        self.legend = legend
        
     
        if LI not in ['LIY', 'LIX']:
            self.is_ok = False
            #log.wrn("Lock-In channes should be 'LIX' or 'LIY'")

        if 'Bias Spectroscopy>Channels' not in self.header:
            # Not a Bias Spec file
            self.is_ok = False
            #log.wrn("%s is not as BiasSpec file.", self.filename)
            return

    @property
    def bias(self):
        return self.data[self.keys.Bias]

    @property
    def dIdV(self):
        return self.data[self.keys.dIdV]

    @property
    def current(self):
        return self.data[self.keys.Current]

    def set_units(self, name, unit):

        _name = self._find_key_for_name(name)

        fullname = self.field_names[_name]
        u1 = fullname.find('[')+1
        u2 = fullname.find(']')
        oldunit = fullname[u1:u2]

        if len(oldunit) > 1:
            self.field_names[_name] = "%s%s%s" % (
                    fullname[:u1], unit, fullname[u1 + 1:])
        else:
            self.field_names[_name] = "%s%s%s" % (
                fullname[:u1], unit, fullname[u1:])
        try:
            del self.data
        except AttributeError:
            pass
        try:
            del self.keys
        except AttributeError:
            pass

    def _find_key_for_name(self, name):
        if name in self.field_names:
            return name
        else:
            for key, fullname in self.field_names.items():
                if fullname == name:
                    return key
        log.err("Field name %s not found.", name)
        return self.field_names.keys()[0]

    @lazy
    def rawdata(self):
        return super().get_data()

    @lazy
    def data(self):
        rk = self.rawkeys
        k = self.keys

        raw = self.rawdata.sort_values(rk.Bias)

        N = raw.shape[0]

        V = raw[rk.Bias]
        I = raw[rk.Current]
        if self.correct_bias_offset is True:
            V -= np.interp(0, I, V)
        elif self.correct_bias_offset is not False:
            V -= self.correct_bias_offset
            
        LI = raw[rk.LI]

        cal = {}

        dV = (V.max() - V.min()) / (N-1)
        cal[k.NdIdV] = np.gradient(I, dV, edge_order=2)

        # limits noise issues
        nl = np.round(self.noise_limits * N).astype(int)
        ratio_numeric = (cal[k.NdIdV] / LI)[nl[0]:nl[1]].mean()

        if self.LI_sens is not None and self.LI_ampl is not None:
            ratio_lockin = self.LI_sens / (2*self.LI_ampl)
        else:
            ratio_lockin = np.NaN

        cal[k.dIdV_FIT] = LI * ratio_numeric
        cal[k.dIdV_LI] = LI * ratio_lockin

        if np.isnan(ratio_lockin) or self.force_numerical_dIdV:
            numerical_dIdV = True
            cal[k.dIdV] = cal[k.dIdV_FIT]
        else:
            numerical_dIdV = False
            cal[k.dIdV] = cal[k.dIdV_LI]

        rkd = rk.__dict__
        kd = k.__dict__

        for key, fullname in self.field_names.items():
            unit = fullname[fullname.find('[')+1:fullname.find(']')]
            if len(unit) > 1 and unit[0] in unit_factor:
                factor = unit_factor[unit[0]]
            else:
                factor = 1.0

            if kd[key] in cal:
                cal[kd[key]] = cal[kd[key]] / factor
            elif key in rkd:
                cal[kd[key]] = raw[rkd[key]] / factor
                
        df = pd.DataFrame(cal)
        
        if self._hamming > 0:
            df[k.dIdV] = df.rolling(
                self._hamming, 
                win_type="hamming", 
                min_periods=0,
                center=True,
            ).mean()[k.dIdV]
        return df

    @lazy
    def keys(self):
        return SimpleNamespace(**self.field_names)

    @lazy
    def rawkeys(self):
        """ Infer the key names from the file header."""
        k = {}
        for key, names in self.search_names.items():
            for name in names:
                if name in self.channel_names:
                    k[key] = name
                    break

        assert 'Current' in k
        assert 'Bias' in k
        assert self.LI in k

        k['LI'] = k[self.LI]

        return SimpleNamespace(**k)

    @lazy_property
    def channel_names(self):
        """ List of channels names that can be found in file."""
        header_names = [s.strip() for s in
                        self.header['Bias Spectroscopy>Channels'].split(';')]

        # 'Bias calc (V)' is in file but not in the header.
        return ['Bias calc (V)', ] + header_names

    def plot(self, title=None, ax=None, save=False, size=None,
             x=None, y=None, xlabel=None, ylabel=None, label=None,
             pyplot=True, dpi=100,
             **kwargs,):
        if 'figsize' not in kwargs and ax is None:
            kwargs['figsize'] = get_figsize(size)

        if x is None:
            x = self.keys.Bias
        if y is None:
            y = self.keys.dIdV

        _x = self.field_names[self._find_key_for_name(x)]
        _y = self.field_names[self._find_key_for_name(y)]

        if xlabel is None:
            xlabel = _x
        if ylabel is None:
            ylabel = _y
        if label is None:
            #label = "%.03d" % self.serie_number
            label = "%s" % self.fn_noext

        if ax is None:
            fig = create_figure(size=size, pyplot=pyplot, dpi=dpi,
                                   shape='golden')
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        ax = self.data.plot(x=_x, y=_y, ax=ax, label=label, **kwargs)

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if self.legend is False:
            ax.legend().remove()

        if save is not False:
            if save is True:
                if title is not None:
                    filename = title + '.png'
                else:
                    filename = self.name + '.png'
            else:
                filename = save
            ax.get_figure().savefig(filename)

        return fig, ax

    @property
    def name(self):
        return self.serie_name + "%.3i" % self.serie_number

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
        if self.filename.endswith(".xz"):
            return os.path.basename(self.filename)[:-10]
        return os.path.basename(self.filename)[:-7]

    @lazy_property
    def serie_number(self):
        if self.filename.endswith(".xz"):
            return int(os.path.basename(self.filename)[-10:-7])
        return int(os.path.basename(self.filename)[-7:-4])

    @lazy_property
    def pixels(self):
        return int(self.header['Bias Spectroscopy>Num Pixel'])
    
    @lazy_property
    def hold(self):
        if self.header['Z-Ctrl hold'] == "FALSE":
            return False
        return True

    @lazy_property
    def info(self):
        return {
            'filename': self.filename,
            'pixels': self.pixels,
            'sweep start': self.v_start,
            'sweep end': self.v_end,
            'datetime': self.datetime
        }
    
    @lazy_property
    def dfentry(self):
        return {
            'path': self.path,
            'pixels': self.pixels,
            'sweep start': self.v_start,
            'sweep end': self.v_end,
            'datetime': self.datetime,
            'hold': self.hold
        }

    def get_base64_plot(self, **kwargs):
        ax = self.plot(pyplot=False, **kwargs)
        bts = io.BytesIO()
        ax.get_figure().savefig(bts, format='png')
        bts.seek(0)
        return base64.b64encode(bts.getvalue()).decode()
