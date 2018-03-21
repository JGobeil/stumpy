""" .dat DataSet. Moved to this file. Not in working condition."""

from .helper.dataset import DataSetBase

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
