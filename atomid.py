import numpy as np
from collections import namedtuple
from matplotlib.patches import Circle
import pandas as pd


class _PtIdConfig(dict):
    class _AtomConfig(list):
        def __init__(self):
            super().__init__()
    
        def add_point(self, bias, value):
            """ Add a point to the config.
    
            Parameters
            ----------
            bias: tuple
                The (min, max) of the bias of the point.
            value: tuple
                The (min, max) associate with this point. How
                this is use depend on the identification method.
            """
            self.append((*bias, *value))
    
        def __repr__(self):
            return "\n".join([
	        "%3d: %s" % (i, s) for i, s in enumerate(self)])
    
    def __init__(self):
        super().__init__()
    
    def create_config(self, atom):
        """ Create a new atom id config. Return the newly created
        config. Erase any existing config with this name.
    
        Parameters
        ----------
        atom: str
            The name of the config.
        
        Returns
        -------
        cfg: The newly create config
        
        """
        self[atom] = _PtIdConfig._AtomConfig()
        return self[atom]


# The Average Id Config database
average_id_config = _PtIdConfig()

# create a new 'type' of atom	
_fe_config = average_id_config.create_config(atom="Fe")
# we can get it (if exist)
_fe_config = average_id_config["Fe"]
# add a point
_fe_config.add_point(bias=(-0.0180, -0.0120), value=(0.6, 1.00))
_fe_config.add_point(bias=(-0.0025, -0.0010), value=(0.0, 0.40))
_fe_config.add_point(bias=(-0.0005, +0.0005), value=(0.0, 0.05))
_fe_config.add_point(bias=(+0.0001, +0.0025), value=(0.0, 0.40))
_fe_config.add_point(bias=(+0.0120, +0.0180), value=(0.6, 1.00))

	
slope_id_config = _PtIdConfig()
_fe_config = slope_id_config.create_config(atom="Fe")
_fe_config.add_point(bias=(-0.0170, -0.0070), value=(0.85, 1.00))
_fe_config.add_point(bias=(-0.0025, +0.0025), value=(0.60, 0.80))
_fe_config.add_point(bias=(+0.0070, +0.0170), value=(0.85, 1.00))



class _AtomId:
    def plot(self, channel, radius=0.8,
             colors=["Green", "Yellow", "Red"],
             with_number=False,
             **kwargs):
        ax = channel.plot(**kwargs)
        topo = channel.topo

        if len(self.identified.shape) == 1:
            identified = np.array([self.identified, self.identified]).T
        else:
            identified = self.identified

        for spec, identified in zip(self.specs, identified):
            xy = topo.abs2topo(spec.xyz_nm[0:2])
            r = radius
            if np.logical_and(*identified):
                c = colors[0]
            elif np.logical_or(*identified):
                c = colors[1]
            else:
                c = colors[2]

            if with_number is False:
                ax.add_patch(Circle(xy, r, fill=False, linewidth=3, color=c))
            elif with_number in [True, "all", "error"]:
                if with_number == "error" and c == colors[0]:
                    continue
                ax.text(xy[0], xy[1], "%.3d" % spec.serie_number,
                        fontdict={'color': c, 'weight': "bold", 'size': 15}
                        )
        return ax


class AverageId(_AtomId):
    def __init__(self, element, specs):
        self.specs = specs
        self.conditions = average_id_config[element]

        self._mins = np.array([c[2] for c in self.conditions])
        self._maxs = np.array([c[3] for c in self.conditions])

        self.identified = []
        self.pts_value = []
        self.pts_ok = []
        for s in specs:
            lockin = s.data[s.keys.LI]
            bias = s.data[s.keys.V]
            pts = np.array([
                lockin[bias.between(c[0], c[1])].mean()
                for c in self.conditions])
            pts -= np.min(pts)
            pts /= np.max(pts)

            pts_ok = (self._mins <= pts) & (pts <= self._maxs)
            self.pts_value.append(pts)
            self.pts_ok.append(pts_ok)
            self.identified.append(np.all(pts_ok))


class SlopeId(_AtomId):
    def __init__(self, element, specs):
        self.specs = specs
        self.conditions = slope_id_config[element]

        self._mins = np.array([c[2] for c in self.conditions])
        self._maxs = np.array([c[3] for c in self.conditions])

        self.identified = []
        self.lines = []
        self.fits = []
        self.fits_ok = []
        for s in specs:
            k_V = s.keys.V
            k_I = s.keys.I

            bias = s.data[s.keys.V]
            lines = [s.data[[k_V, k_I]][bias.between(c[0], c[1])]
                     for c in self.conditions]

            fits = np.array([np.polyfit(line[k_V], line[k_I], deg=1)
                             for line in lines])[:, 0]
            fits /= np.max(fits)
            fits_ok = (self._mins <= fits) & (fits <= self._maxs)

            self.lines.append(lines)
            self.fits.append(fits)
            self.fits_ok.append(fits_ok)
            self.identified.append(np.all(fits_ok))


class DoubleId(_AtomId):
    def __init__(self, element, specs):
        self.specs = specs
        self.average_id = AverageId(element, specs)
        self.slope_id = SlopeId(element, specs)

        self.identified = np.array([
            self.average_id.identified,
            self.slope_id.identified]).T
