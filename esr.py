import numpy as np

from .datfile import GenericDatFile
from .helper import lazy_property


class Zspec(GenericDatFile):

    @lazy_property
    def z_offset_nm(self):
        return float(self.header['Z offset (m)'])*1e9

    @lazy_property
    def z_sweep_distance_nm(self):
        return float(self.header['Z sweep distance (m)'])*1e9

    @lazy_property
    def settling_time_s(self):
        return float(self.header['Settling time (s)'])

    @lazy_property
    def integration_time_s(self):
        return float(self.header['Integration time (s)'])

    @lazy_property
    def data(self):
        data = super().data

        data['Z rel (nm)'] = data['Z rel (m)']*1e9
        data['Z (nm)'] = data['Z (m)']*1e9
        if 'Current (A)' in data:
            data['Current (nA)'] = data['Current (A)']*1e9
            if 'Lock-In R (V)' in data:
                data['LI R / Current (V/nA)'] = data['Lock-In R (V)'] / data['Current (nA)']
            if 'Lock-In X (V)' in data:
                data['LI X / Current (V/nA)'] = data['Lock-In X (V)'] / data['Current (nA)']
            if 'Lock-In Y (V)' in data:
                data['LI Y / Current (V/nA)'] = data['Lock-In Y (V)'] / data['Current (nA)']

        return data

