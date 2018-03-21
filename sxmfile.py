"""Open and parse an sxm file"""

import hashlib

import numpy as np
import pandas as pd
from io import StringIO
from collections import ChainMap

from .helper import get_logger
from .helper.fileparser import ColonHeaderFile
from .helper.fileparser import Parse
from .helper.lazy import lazy_property
from .plotting import create_figure, add_title
from .plotting import no_grid, no_axis, no_ticks

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

    def __init__(self, filename, common_path=None):
        """ Open a sxm file.

        Parameters
        ----------
        filename : str
            name of the file to open
        use_cache: bool
            if true use memory cache for the channels
        """
        super().__init__(filename, common_path)

        if self.is_ok:
            self.channels = {i: SxmChannel(i, self) for i
                             in range(self.number_of_channels)}

    def plot(self, ax=None, channel=None, **kwargs):
        if channel is None:
            channel = 0
        return self.channels[channel].plot(ax=ax, **kwargs)

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
        return "%.5g nm" % nmx if nmx == nmy else "%.5gx%.5g nm" % (nmx, nmy)

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
    def rotation_matrix(self):
        """ Rotation matrix using the angle of the scan."""
        theta = np.radians(-self.angle)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

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
    def current_nA(self):
        """ Scanning current in nA"""
        return float(self.z_controller[0]['Setpoint'].split()[0]) * 1e12

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
        return pd.read_table(StringIO(self.header['Multipass-Config']))


# Modification and plotting of sxm channel
class SxmChannel:
    def __init__(self, channel_number: int, sxm: SxmFile):

        """ Get the raw numpy array of a channel."""
        self.number = channel_number
        self.name = sxm.channel_names[channel_number]
        self.sxm = sxm
        self.plot = _SxmChannelPlot(self)

        self.info = {
            'name': self.name,
            'bias': sxm.bias,
            'current_nA': sxm.current_nA,
            'size_px': sxm.size_px,
            'size_nm': sxm.size_nm,
        }


    @lazy_property
    def data(self):
        sxm = self.sxm
        with sxm.file as f:
            f.seek(sxm.datastart + self.number * sxm.chunk_size)
            raw = f.read(sxm.chunk_size)
            data = np.frombuffer(raw, dtype='>f4').reshape(*sxm.shape)

        if sxm.direction == 'up':
            data = np.flipud(data)
        if self.number & 0x1:
            data = np.fliplr(data)
        return data

    def subtract_average(self, direction=None):
        return SxmChannel_SubtractAverage(self, direction=direction)


class SxmChannel_SubtractAverage(SxmChannel):
    def __init__(self, src: SxmChannel, direction=None):
        """ Subtract Average of each line (default) of columns.

        Parameters
        ----------
        direction: str
            Direction in with to subtract average. None or 'x' subtract
            average in the 'horizontal' direction'. 'y' for subtraction in
            the 'vertical' direction'.
        """
        self.src = src
        self._direction = 0
        self.direction = direction
        super().__init__(src.number, src.sxm)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        if value in [None, 'x']:
            self._direction = 1
        elif value in ['y', ]:
            self._direction = 0
        else:
            log.err('Not a valid direction: %s' % value)

    @lazy_property
    def data(self):
        data = super().data
        return data - np.mean(data, axis=0)


class _SxmChannelPlot:
    def __init__(self, sxmchannel: SxmChannel):
        self.channel = sxmchannel
        self.sxm = sxmchannel.sxm

    def __call__(
            self,
            ax=None,
            show_axis=False,
            size='normal',
            info='normal',
            cmap='Blues_r',
            boxed=True,
            pyplot=True,
            dpi=100,  # only when not using pyplot
            **kwargs
            ):
        cnl = self.channel
        sxm = self.sxm

        if ax is None:
            figure = create_figure(size=size, pyplot=pyplot, dpi=dpi)
            ax = figure.add_subplot(111)

        x1, y1 = (0, 0)
        x2, y2 = np.array([x1, y1]) + sxm.size_nm

        ax.imshow(
            cnl.data,
            cmap=cmap,
            extent=(x1, x2, y1, y2),
            **kwargs
        )

        if info == "normal":
            infostr = "%s\n%s\n%s/%s - %gpA@%s\n%s" % (
                sxm.path,  # filename
                sxm.record_datetime,
                sxm.size_nm_str,  # size
                sxm.size_px_str,  # pixels
                sxm.current_nA,  # current,
                sxm.bias_str,  # bias
                sxm.name)
        elif info == "minimal":
            infostr = "%.3d - %s - %s" % (
                sxm.serie_number,  # filename
                sxm.size_nm_str,  # size
                sxm.bias_str,  # bias
                )
        elif info is None:
            infostr = ""
        else:
            infostr = info
        add_title(ax, infostr)

        if not show_axis:
            no_axis(ax)
        if boxed:
            no_ticks(ax)
        no_grid(ax)
        return ax

    def save(self, filename, **kwargs):
        # TODO: Add auto filename generation
        self(**kwargs, pyplot=False).get_figure().savefig(filename)
        return filename
