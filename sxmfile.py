"""Open and parse an sxm file"""

import numpy as np
import pandas as pd
from collections import ChainMap
import io
import re

from .helper import get_logger
from .helper.fileparser import ColonHeaderFile
from .helper.fileparser import Parse
from .helper.lazy import lazy_property
from .topo import Topo
from .topo import TopoSet

from . import defaults

log = get_logger(__name__)

__author__ = 'Jeremie Gobeil'
__version__ = '2.0'

_multipass_re = re.compile(".*\[P([0-9]+)\].*")


class SxmFile(ColonHeaderFile):
    """ An .sxm file with parsed header and on-demand channels data.

    When a new object is created only the header is read an kept in
    memory. The channels data will be read and kept in memory when
    accessing the channel data.
    """
    header_end = ':SCANIT_END:'
    dataoffset = 4

    def __init__(self, filename, plot_defaults=None, channels=None):
        """ Open a sxm file.

        Parameters
        ----------
        filename : str
            name of the file to open
        plot_defaults: dict
            default configuration for the plot
        """
        filename = str(filename)
        if len(filename) > 3 and not (filename.endswith(".sxm") or filename.endswith(".sxm.xz")):
            filename = filename + ".sxm"
        super().__init__(filename)

        self.plot_defaults = ChainMap(defaults['topoplot']).new_child()
        if plot_defaults is not None:
            self.plot_defaults.update(plot_defaults)

        if self.is_ok:
            if channels is None:
                channels = defaults['channels']
            self.channels = TopoSet([Topo(SxmChannel(i, self)) for i in
                                     range(self.number_of_channels)]
                                    ).filter_by_name(channels)

    def __getitem__(self, item):
        return self.channels[item]

    def set_plot_defaults(self, **kwargs):
        """ Set new value for the default plots"""
        self.plot_defaults.update(**kwargs)

    def plot(self, channel=None, ax=None, **kwargs):
        if channel is None:
            return self.channels.plot(**kwargs)
        else:
            return self.channels[channel].plot(ax=ax, **kwargs)

    @lazy_property
    def size_nm_str(self):
        nmx, nmy = self.size_nm
        return "%.5g nm" % nmx if nmx == nmy else "%.5gx%.5g nm" % (nmx, nmy)

    @lazy_property
    def size_px_str(self):
        pxx, pxy = self.size_px
        return "%d px" % pxx if pxx == pxy else "%dx%d px" % (pxx, pxy)

    def __repr__(self):
        if self.is_ok:
            return "SxmFile - '%s' - [%s-%s]" % (
                self.filename, self.size_nm_str, self.size_px_str)
        else:
            return "SxmFile - Error opening file '%s'." % (
                self.filename)


    @lazy_property
    def info(self):
        return {
            'path': self.filename,
            'obj': self,
            'name': self.name,
            'datetime': self.record_datetime,
            'bias': self.bias,
            'size_nm': self.size_nm[0],
            'size_px': self.size_px[0],
            'size_ratio': self.size_nm[0] / self.size_nm[1],
            'nb_channels': self.number_of_channels,
            'pos_x_nm': self.pos_nm[0],
            'pos_y_nm': self.pos_nm[1],
            'serie_name': self.serie_name,
            'serie_number': self.serie_number,
            'Current (nA)': self.current_nA,
            #'Z (m)': self.Z_m,
        }
    
    @lazy_property
    def dfentry(self):
        return{
            'path': self.path,
            'datetime': self.record_datetime,
            'bias': self.bias,
            'size_nm^2': self.size_nm[0]*self.size_nm[1],
            'size_px': self.size_px[0]*self.size_px[1],
            'nb_channels': self.number_of_channels,
            'Current (nA)': self.current_nA,
            "is_multipass": self.is_multipass,
        }

    @lazy_property
    def original_filename(self):
        return self.header['SCAN_FILE']

    @lazy_property
    def serie_name(self):
        """The serie name"""
        return self.header['Scan>series name']

    @lazy_property
    def serie_number(self):
        """The sequential number within the serie."""
        sn = self.serie_name
        of = self.original_filename
        return int(of[of.rfind(sn) + len(sn):-4])

    @lazy_property
    def name(self):
        "Name of the sxmfile, i.e. serie name + number."
        return "%s%.3d" % (self.serie_name, self.serie_number)

    @lazy_property
    def record_datetime(self):
        """ The Datetime object corresponding the the record time"""
        return Parse.datetime(self.header['REC_DATE'], self.header['REC_TIME'])

    @lazy_property
    def datetime(self):
        """ The Datetime object corresponding the the record time"""
        return self.record_datetime

    @lazy_property
    def acquisition_time_s(self):
        """ The time it took for acquisition"""
        return float(self.header['ACQ_TIME'])

    @lazy_property
    def size_px(self):
        """ Size of the scan in pixels (np.array with shape [2,])"""
        return np.fromstring(
            self.header['SCAN_PIXELS'],
            dtype=int,
            sep=' ')

    @lazy_property
    def size_nm(self):
        """ Size of the scan in nm (np.array with shape [2,])"""
        return 1e9 * np.fromstring(
            self.header['SCAN_RANGE'],
            dtype=float,
            sep=' ',
        )

    @lazy_property
    def bias(self):
        return float(self.header['BIAS'])

    @lazy_property
    def bias_str(self):
        b = self.bias
        return "%.3g mV" % (b * 1000) if b < 1 else "%.3g V" % b

    @lazy_property
    def resolution(self):
        """ pixels per nm (np.array with shape [2,]"""
        return self.size_px / self.size_nm
    	
    @lazy_property
    def pixels(self):
        return np.prod(self.size_px)

    @lazy_property
    def scan_speed_nm_s(self):
        """ Scan speed in nm/s"""
        return float(self.header['Scan>speed forw. (m/s)']) * 1e9

    @lazy_property
    def direction(self):
        """ Direction of the scan. 'up' or 'down'"""
        return self.header['SCAN_DIR']

    @lazy_property
    def pos_nm(self):
        """ Position of the scan in nm (np.array with shape [2,]"""
        return 1e9 * np.fromstring(
            self.header['SCAN_OFFSET'],
            dtype=float,
            sep=' ')

    @lazy_property
    def angle(self):
        """ Scanning angle"""
        return float(self.header['SCAN_ANGLE'])

    @lazy_property
    def data_info(self):
        """ Table data info"""
        return Parse.table(self.header['DATA_INFO'],
                           int, str, str, str, float, float)

    @lazy_property
    def channel_names(self):
        """ The names of the channels with -fwd or -bwd if the scan is in
        both direction"""
        channel_names = []
        for channel in self.data_info:
            name = channel['Name']
            direction = channel['Direction']
            if direction == 'both':
                channel_names.extend([name + '-fwd', name + '-bwd'])
            else:
                channel_names.append(name + '-' + direction)
        return channel_names

    @lazy_property
    def channel_number(self):
        """ The number of the channels """
        channel_number = {}
        for i in range(self.number_of_channels):
            channel_number[i] = i
            channel_number[self.channel_names[i]] = i
        return channel_number

    @lazy_property
    def number_of_channels(self):
        return sum([2 if channel['Direction'] == 'both' else 1
                    for channel in self.data_info])

    @lazy_property
    def comment(self):
        return self.header['COMMENT']

    @lazy_property
    def z_controller(self):
        """ Z-controller info"""
        return Parse.table(
            self.header['Z-CONTROLLER'],
            str, bool, str, str, str, str)

    @lazy_property
    def current_pA(self):
        """ Scanning current in pA"""
        return self.current_A * 1e12

    @lazy_property
    def current_nA(self):
        """ Scanning current in nA"""
        return self.current_A * 1e9

    @lazy_property
    def current_A(self):
        """ Scanning current in nA"""
        return float(self.z_controller[0]['Setpoint'].split()[0])

    #@lazy_property
    #def Z_m(self):
    #    """ Scanning current in nA"""
    #    return float(self.header['Z-Controller>Z (m)'])

    @lazy_property
    def shape(self):
        return tuple(self.size_px[::-1])

    @lazy_property
    def chunk_size(self):
        return 4 * self.size_px.prod()

    @lazy_property
    def is_multipass(self):
        """ Is the scan is multipass? """
        return 'Multipass-Config' in self.header

    @lazy_property
    def multipass_config(self):
        """ The multipass configuration in pandas.DataFrame format."""
        return pd.read_csv(io.StringIO(self.header['Multipass-Config']), sep='\t')


class SxmChannel:
    """ Channel loader"""
    def __init__(self, name_or_number: int, sxm: SxmFile):

        """ Get the raw numpy array of a channel."""
        self.number = name_or_number
        self.name = sxm.name + '.' + sxm.channel_names[name_or_number]
        self.sxm = sxm
        self.channel = self
        self.number = name_or_number
        self.name = sxm.name + '.' + sxm.channel_names[name_or_number]
        self.direction = self.name[-3:]

        self.bias = sxm.bias
        self.current = sxm.current_nA
        self.scan_speed_nm_s = sxm.scan_speed_nm_s

        if sxm.is_multipass:
            # #TODO: add other multipass info (speed, ...)
            P = _multipass_re.findall(self.name)
            if len(P) > 1:
                log.wrn("Found multiple pass in when parsing %s."
                        "Using P=%s", self.name, P[-1])
                P = int(P[-1])
            elif len(P) == 1:
                P = int(P[-1])
            else:
                P = 1

            N = (P-1) * 2 + (1 if self.direction == 'bwd' else 0)
            mp = sxm.multipass_config.iloc[N]
            if mp['Bias_override']:
                self.bias = mp['Bias_override_value']
            if mp['Z_Setp_override']:
                self.current = mp['Z_Setp_override_value'] * 1e9
            self.scan_speed_nm_s *= mp['Speed_factor']


    @property
    def current_nA(self):
        return self.current

    @property
    def current_pA(self):
        return self.current * 1000

    @lazy_property
    def data(self):
        """ Numpy array of the data. For indice is x the second is y
        so data[0, 1] give the pixels above the bottom left corner. """
        sxm = self.sxm
        with sxm.file as f:
            f.seek(sxm.datastart + self.number * sxm.chunk_size)
            raw = f.read(sxm.chunk_size)
            data = np.frombuffer(raw, dtype='>f4').reshape(*sxm.shape)

        if sxm.direction == 'down':
            data = np.flipud(data)
        if self.number & 0x1:
            data = np.fliplr(data)
        return data.T

    @lazy_property
    def bias_str(self):
        b = self.bias
        return "%.3g mV" % (b * 1000) if b < 1 else "%.3g V" % b

    @lazy_property
    def current_str(self):
        c = self.current
        return "%.3g pA" % (c * 1000) if c < 1 else "%.3g nA" % c


    def __getattr__(self, name):
        """ If the attribute don't exist, try the sxm attribute."""
        return self.sxm.__getattribute__(name)

