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
