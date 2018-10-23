""" Module to find atoms coordinate in a topo scan. """

#from types import SimpleNamespace
#import pandas as pd
from .topo2 import Topo
from .lazy import lazy_property



conf_default = _conf_fe


class AtomSearch:
    def __init__(self, conf=None):

        self.conf = atom_search_default.copy()

        if conf != None:
            self.conf.update(**conf)

    def __call__(self, topo: Topo, conf=None):
        if conf is None:
            conf = self.conf




