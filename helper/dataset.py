""" DataSetBase helper class"""

import os.path
from glob import glob

import pandas as pd

from . import get_logger
from . import lazy_property

log = get_logger(__name__)


class DataSetBase:
    def __init__(self, *things, cls, opener, ext,
                 sort_key=None, index_key=None):

        self.sort_key = sort_key
        self.index_key = index_key

        self._objs = set()
        files = []

        for thing in things:
            if isinstance(thing, cls):
                self._objs.add(thing)
            elif isinstance(thing, self.__class__):
                self._objs.update(thing.objs)
            elif isinstance(thing, pd.DataFrame):
                self._objs.update(thing.obj)
            elif os.path.isdir(thing):
                files.extend(glob(os.path.join(thing, '*' + ext)))
            else:
                files.append(thing)

            if files:
                common_path = os.path.commonpath(files)

                self._objs.update([opener(
                    os.path.relpath(path, common_path),
                    common_path=common_path) for path in files])

    @lazy_property
    def data(self):
        return self.get_data()

    def get_data(self):
        data = pd.DataFrame([obj.info for obj in self._objs])
        if self.sort_key is not None:
            data.sort_values(self.sort_key, inplace=True)
        if self.index_key is not None:
            data.set_index(self.index_key, inplace=True)
        return data

    @property
    def objs(self):
        return self.data['obj']

    def merge(self, other, analyse=True):
        self._objs = self._objs.union(other._specs)
        del self.data

    def __repr__(self):
        return '<%s with %i entries>' % (
            self.__class__.__name__,
            len(self.objs),
        )

    def __len__(self):
        return len(self.data)


    def get(self, *things, basename=None):
        out = []
        for t in things:
            if isinstance(int, t):
                out.append()


#from glob import glob
#from os.path import join as pjoin
#from types import SimpleNamespace
#
#
#class DataGroup:
#    def __init__(self, cache=None):
#        if cache is None:
#            self._cache = {}
#        else:
#            self._cache = cache
#
#        self.sxms = None
#
#    def add(*files):
#        sxms = [f for f in files is f[-3:] == 'sxm']
#        if len(sxms) > 0:
#            if self.sxms is None:
#                self.sxms = SxmDataGroup()
#            self.sxms.add(*sxms)
#
#
#class SxmDataGroup(DataGroup):
#    def __init__(self, cache=None):
#        super().__init__(cache=cache)
#        self.series = {}
#
#    def add(*files):
#        log = SimpleNamespace(
#            total=0,
#            cache_hit=0,
#            new_basename=[],
#            bad_sxm=0,
#        )
#
#    for file in files:
#        if file in self._cache:
#            out.cache_hit += 1
#            sxm = self._cache
#        else:
#            sxm = SxmFile(file)
#
#        if not sxm.is_ok:
#            out.bad_sxm += 1
#            continue
#
#        if sxm.serie_name not in self.series:
#            self.series.append(sxm.serie_name)
#            out.new_series.append(sxm.serie_name)
#
#
#class DataPath(DataGroup):
#    def __init__(self, *paths, cache=None):
#        self.paths = paths
#        if cache is None:
#            self._cache = {}
#        else:
#            self._cache = cache
#
#        self.series = {}
#
#    def read_paths(self):
#        for path in paths:
#            self._read_path(path)
#
#    def _read_path(self, path):
#        self._read_sxms(path)
#
#    def _read_sxms(self, path):
#        out = SimpleNamespace(
#            total=0,
#            cache_hit=0,
#            new_basename=[],
#            bad_sxm=0,
#        )
#        files = glob(pjoin(os.path, '*.sxm'))
#
#        for file in files:
#            if file in self._cache:
#                out.cache_hit += 1
#                sxm = self._cache
#            else:
#                sxm = SxmFile(file)
#
#            if not sxm.is_ok:
#                out.bad_sxm += 1
#                continue
#
#            if sxm.serie_name not in self.series:
#                self.series.append(sxm.serie_name)
#                out.new_basename.append(sxm.serie_name)

