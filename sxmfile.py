"""Open and parse an sxm file"""

import hashlib

import matplotlib.pyplot as plt
import numpy as np

from .helper import get_logger
from .helper.fileparser import ColonHeaderFile
from .helper.fileparser import Parse
from .helper.lazy import lazy_property
from .topo.topo import Topo
from .topo.topo import topomod

log = get_logger(__name__)

__author__ = 'Jeremie Gobeil'
__version__ = '1.3'


class SxmFile(ColonHeaderFile):
    """ An .sxm file with parsed header and on-demand channels data.

    When a new object is created only the header is read an kept in
    memory. The channels data will be read and kept in memory when
    accessing the channel data.

    #TODO: Parsing of scan with different forward and backward settings
    """
    header_end = ':SCANIT_END:'
    dataoffset = 4

    def __init__(self, filename, common_path=None, use_cache=True):
        """ Open a sxm file.

        Parameters
        ----------
        filename : str
            name of the file to open
        use_cache: bool
            if true use memory cache for the channels
        """
        super().__init__(filename, common_path)

        self.use_cache = use_cache

        if self.is_ok:
            self.channels = SxmFile.ChannelLoader(self)


    def plot(self, ax=None, channel=None, show_axis=False, figsize=(10, 10)):
        if channel is None:
            channel = 0
        return SxmChannel(channel=channel, sxm=self).plot(
            ax=ax,
            show_axis=show_axis,
            figsize=figsize,
        )

    @lazy_property
    def uid(self):
        """An unique identifier for the scan. It as the format
        'SXM-YYYYMMDD-HHMMSS-xxx' where xxx is the first 3 value of the
        md5 sum of the original filename."""
        return '-'.join([
            'SXM',
            self.record_datetime.strftime('%Y%m%d-%H%M%S'),
            hashlib.md5(self.original_filename.encode()).hexdigest()[0:3],
        ])

    @lazy_property
    def size_nm_str(self):
        nmx, nmy = self.size_nm
        return "%g nm" % nmx if nmx == nmy else "%gx%g nm" % (nmx, nmy)

    @lazy_property
    def size_px_str(self):
        pxx, pxy = self.size_px
        return "%d px" % pxx if pxx == pxy else "%dx%d px" % (pxx, pxy)

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        return "SxmFile - '%s' - [%s-%s]" % (
            self.filename, self.size_nm_str, self.size_px_str)

    @lazy_property
    def info(self):
        return {
            'path': self.filename,
            'obj': self,
            'uid': self.uid,
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
        return int(self.original_filename[-7:-4])

    @lazy_property
    def name(self):
        return "%s%.3d" % (self.serie_name, self.serie_number)

    @lazy_property
    def record_datetime(self):
        """ The Datetime object corresponding the the record time"""
        return Parse.datetime(self.header['REC_DATE'], self.header['REC_TIME'])

    @lazy_property
    def acquisition_time_s(self):
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
        names = []
        for channel in self.data_info:
            name = channel['Name']
            if channel['Direction'] == 'both':
                names.extend((name + '-fwd', name + '-bwd'))
            else:
                names.append('%s (%s)' % (name, channel['Direction']))
        return names

    @lazy_property
    def channel_number(self):
        """ The number of the channels """
        channel_number = {}
        for i in range(len(self.channel_names)):
            channel_number[i] = i
            channel_number[self.channel_names[i]] = i

        return channel_number

    @lazy_property
    def number_of_channels(self):
        return len(self.channel_names)

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
    def current_nA(self):
        """ Scanning current in nA"""
        return float(self.z_controller[0]['Setpoint'].split()[0]) * 1e12

    @lazy_property
    def shape(self):
        return tuple(self.size_px[::-1])

    @lazy_property
    def chunk_size(self):
        return 4 * self.size_px.prod()

    class ChannelLoader:
        """ Class to load the channels on demand"""

        def __init__(self, sxm, use_cache=True):
            self.sxm = sxm

            # to read data
            self.datastart = sxm.datastart
            self.direction = sxm.direction

            # channel names
            self.names = sxm.channel_names
            self.channel_number = sxm.channel_number

            # dictionary to find channel number by it number or it name

            # channel data caching
            self.use_cache = use_cache
            self._channel_cache = {}

        def __len__(self):
            return len(self.names)

        def __getitem__(self, number_or_name):
            return SxmChannel(number_or_name, self)

        def get_data(self, number_or_name):
            """ Get the raw numpy array of a channel."""
            n = self.channel_number[number_or_name]
            sxm = self.sxm

            # use the cached value if available
            if self.use_cache and n in self._channel_cache:
                return self._channel_cache[n]

            with self.sxm.file as sxm:
                sxm.seek(self.datastart + n * self.chunk_size)
                raw = sxm.read(self.chunk_size)
                data = np.frombuffer(raw, dtype='>f4').reshape(*self.shape)

            if self.direction == 'up':
                data = np.flipud(data)
            if n & 0x1:
                data = np.fliplr(data)

            if self.use_cache:
                self._channel_cache[n] = data
            return data

        def __delitem__(self, number_or_name):
            n = self.channel_number[number_or_name]

            if n in self._channel_cache:
                del self._channel_cache[n]

        def reset_cache(self):
            for i in range(len(self)):
                del self[i]


@topomod(topo_method=False)
def from_sxm(sxm, number_or_name):
    return Topo(
        data=sxm.channels.get_data(number_or_name),
        size_nm=sxm.size_nm,
        pos_nm=sxm.pos_nm,
    )


class SxmChannel:
    def __init__(self, channel, sxm: SxmFile):

        """ Get the raw numpy array of a channel."""
        self.number = sxm.channel_number[channel]
        self.name = sxm.channel_names[self.number]
        self.sxm = sxm

        self.size_nm = sxm.size_nm
        self.filename = sxm.filename
        self.path = sxm.path
        self.record_datetime = sxm.record_datetime
        self.size_nm_str = sxm.size_nm_str
        self.size_px_str = sxm.size_px_str
        self.current_nA = sxm.current_nA
        self.bias_str = sxm.bias_str

        with sxm.file as f:
            f.seek(sxm.datastart + self.number * sxm.chunk_size)
            raw = f.read(sxm.chunk_size)
            self.data = np.frombuffer(raw, dtype='>f4').reshape(*sxm.shape)

        if sxm.direction == 'up':
            self.data = np.flipud(self.data)
        if self.number & 0x1:
            self.data = np.fliplr(self.data)

    def plot(self, ax=None, show_axis=False, figsize=(10, 10)):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        x1, y1 = (0, 0)
        x2, y2 = np.array([x1, y1]) + self.size_nm

        ax.imshow(
            self.data,
            cmap='Blues_r',
            extent=(x1, x2, y1, y2)
        )

        if not show_axis:
            ax.axis('off')

        ax.grid(False)

        ax.set_title("%s\n%s\n%s/%s - %gpA@%s\n%s" % (
            self.path,  # filename
            self.record_datetime,
            self.size_nm_str,  # size
            self.size_px_str,  # pixels
            self.current_nA,  # current,
            self.bias_str,  # bias
            self.name,
        ))
        return ax

    def save_plot(self, filename, show_axis=False, figsize=(10, 10)):
        self.plot().ax.get_figure(
            show_axis=show_axis, figsize=figsize).savefig(filename)
