import numpy as np
from collections import namedtuple
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
import pandas as pd
from stumpy.plotting import no_axis, no_ticks, no_grid

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
_fe_config.add_point(bias=(+0.0010, +0.0025), value=(0.0, 0.40))
_fe_config.add_point(bias=(+0.0120, +0.0180), value=(0.6, 1.00))


slope_id_config = _PtIdConfig()
_fe_config = slope_id_config.create_config(atom="Fe")
_fe_config.add_point(bias=(-0.0170, -0.0070), value=(0.85, 1.00))
_fe_config.add_point(bias=(-0.0025, +0.0025), value=(0.60, 0.80))
_fe_config.add_point(bias=(+0.0070, +0.0170), value=(0.85, 1.00))



class _AtomId:
    def __init__(self, specs):
        self.methods = None
        self.specs = specs

    @property
    def is_multimethods(self):
        return self.methods is not None

    def plot(self, channel, radius=0.8,
             colors=["Green", "Yellow", "Red"],
             with_number=False,
             **kwargs):
        ax = channel.plot(**kwargs)
        topo = channel.topo

        if self.is_multimethods:
            identified = self.identified
        else:
            identified = np.array([self.identified, self.identified]).T

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

    def plot_results(self, boxcolor='0.9'):
        if self.is_multimethods:
            methods = self.methods
        else:
            methods = [self, ]

        ncols = len(methods) + 2
        figs = []

        for i, s in enumerate(self.specs):
            fig, axis = plt.subplots(ncols=ncols, figsize=(15, 5))

            if not self.is_multimethods:
                for c in methods[0].conditions:
                    axis[0].axvspan(c[0], c[1], color='0.8', alpha=0.5)

            s.data.plot(s.keys.V, s.keys.dIdV_tilt_corrected, ax=axis[0])
            axis[0].set_title(s.serie_number)
            axis[0].legend().remove()

            for j, mtd in enumerate(methods):
                conditions = mtd.conditions
                #points = [(c[0]+c[1])/2 for c in conditions]
                colors = ['Green' if pt else 'Red' for pt in mtd.pts_ok[i]]

                for k, c in enumerate(conditions):
                    #axis[j+1].hlines(c[2], i, i+1, linestyle='dotted')
                    #axis[j+1].hlines(c[3], i, i+1, linestyle='dotted')
                    axis[j+1].fill_between(
                        [k, k+1],
                        [c[3], c[3]],
                        [c[2], c[2]],
                        color=boxcolor,
                    )

                xp = np.arange(len(conditions)) + 0.5
                axis[j+1].scatter(xp, mtd.pts_value[i], color=colors)

                axis[j+1].set_ylim(-0.1, 1.1)

            axis[ncols-1].text(0.5, 0.5,
                               "\n".join([str(m.identified[i])
                                          for m in methods]),
                               fontsize=18,
                               )
            no_ticks(axis[ncols-1])
            no_axis(axis[ncols-1])

            figs.append(fig)
        return figs


class AverageId(_AtomId):
    def __init__(self, element, specs):
        super().__init__(specs)

        self.conditions = average_id_config[element]

        self._mins = np.array([c[2] for c in self.conditions])
        self._maxs = np.array([c[3] for c in self.conditions])

        self.identified = []
        self.pts_value = []
        self.pts_ok = []
        for s in specs:
            dIdV = s.data[s.keys.dIdV_tilt_corrected]
            bias = s.data[s.keys.V]
            pts = np.array([
                dIdV[bias.between(c[0], c[1])].mean()
                for c in self.conditions])
            pts -= np.min(pts)
            pts /= np.max(pts)

            pts_ok = (self._mins <= pts) & (pts <= self._maxs)
            self.pts_value.append(pts)
            self.pts_ok.append(pts_ok)
            self.identified.append(np.all(pts_ok))
        self.identified = np.array(self.identified)


class SlopeId(_AtomId):
    def __init__(self, element, specs):
        super().__init__(specs)

        self.conditions = slope_id_config[element]

        self._mins = np.array([c[2] for c in self.conditions])
        self._maxs = np.array([c[3] for c in self.conditions])

        self.identified = []
        self.lines = []
        self.pts_value = []
        self.pts_ok = []
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
            self.pts_value.append(fits)
            self.pts_ok.append(fits_ok)
            self.identified.append(np.all(fits_ok))
        self.identified = np.array(self.identified)

class DoubleId(_AtomId):
    def __init__(self, element, specs):
        self.specs = specs
        self.average_id = AverageId(element, specs)
        self.slope_id = SlopeId(element, specs)

        self.methods = [self.average_id, self.slope_id]

        self.identified = np.array([
            self.average_id.identified,
            self.slope_id.identified]).T
