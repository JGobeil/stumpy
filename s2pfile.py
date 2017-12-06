""" Classes and functions to deal with .s2p files from Network analyser."""
import pandas as pd
from matplotlib.style import context as mpl_context

from .helper import get_logger

log = get_logger(__name__)


class TFDataSet:
    """A Transfer functions data set, i.e. multiple transfer function
    in one object. All data are kept as a pandas.DataFrace attribute
    named df. """

    def __init__(self):
        self.df = pd.DataFrame(columns=['GHz'])

    def add_tf(self, filename, tfs=None):
        """ Add a transfer function to the data set.

        Parameters
        ----------
        filename: str
            file from where to load the data.
        tfs: None or str or list or dict
            If None, load S11, S12, S21 and S22. If a str or list of str
            load those specific columns. If a dict, load the columns from
            the keys and rename the columns to the values.
        """
        data = open_s2p(filename)

        if tfs is None:
            c = data.columns
            _tfs = [tf for tf in ['S11', 'S12', 'S21', 'S22'] if tf in c]
            _names = [None, ] * len(_tfs)
        elif isinstance(tfs, str):
            _tfs = [tfs, ]
            _names = [None, ] * len(_tfs)
        elif isinstance(tfs, dict):
            _tfs = tfs.keys()
            _names = tfs.values()
        else:
            _tfs = tfs
            _names = [None, ] * len(_tfs)

        for tf, name in zip(_tfs, _names):
            if name in self.df.columns:
                log.wrn("Name %s already present in the data set", name)
            if tf in self.df.columns:
                log.wrn("Name %s already present in the data set", tf)

            self.df = pd.merge(self.df, data[['GHz', tf]],
                               on='GHz', how='outer')
            if name is not None:
                self.df.rename(columns={tf: name}, inplace=True)
        self.df.sort_values('GHz', inplace=True)

    def plot(self, channels=None, title=None, ax=None, save=False, **kwargs):
        """ Plot transfer functions

        Parameters
        ----------
        channels: str or list of str
            The name(s) of the channel(s) to plot. If None, plot all channels.
        title: str
            Title of the plot. Default: no title
        ax: axis
            matplotlib axis to use. If None, create a new figure.
        save: bool or str
            if not False, save the figure to disk. If save=True use title.png,
            else use save as the filename.

        """
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (10, 7)

        if channels is None:
            channels = [c for c in self.df.columns if c is not 'GHz']

        with mpl_context('ggplot'):
            ax = self.df.interpolate().plot(x='GHz', y=channels,
                                            title=title,
                                            ax=ax,
                                            **kwargs
                                            )
            ax.set_ylabel('dB'),
            if save is not False:
                if save == True:
                    filename = title + '.png'
                else:
                    filename = save
                ax.get_figure().savefig(filename)
            return ax


def open_s2p(filename):
    """ Open a .s2p file (or .s1p) file. Return a pandas.Dataframe."""
    if filename[-4:] == '.s2p':
        usecols = [0, 1, 2, 3, 4]
        names = ['Hz', 'S11', 'S21', 'S12', 'S22', '0', '1', '2', '3', '4']
    else:
        usecols = [0, 1]
        names = ['Hz', 'S11', 'S21']

    data = pd.read_table(filename,
                         skiprows=13,
                         delimiter='\t',
                         header=0,
                         usecols=usecols,
                         names=names,
                         )
    data['GHz'] = data['Hz'] / 1e9
    log("Loaded '%s'. %i rows.", filename, len(data))
    return data
