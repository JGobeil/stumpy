import numpy as np
from collections import namedtuple
from matplotlib.patches import Circle

PtAverageId = namedtuple("PtAverageId", ['left', 'right', 'min', 'max'])


AverageId_default = {
        "Fe": [PtAverageId(l, r, mi, ma) for l, r, mi, ma in [
            (-0.0180, -0.0120, 0.6, 1.0),
            (-0.0025, -0.0010, 0.0, 0.4),
            (-0.0005, +0.0005, 0.0, 0.05),
            (+0.0001, +0.0025, 0.0, 0.4),
            (+0.0120, +0.0180, 0.6, 1.0),
            ]]
        }


class AverageId:
    def __init__(self, element, specs):
        self.specs = specs
        self.conditions = AverageId_default[element]

        self._mins = np.array([c.min for c in self.conditions])
        self._maxs = np.array([c.max for c in self.conditions])

        self.identified = []
        self.pts_value = []
        self.pts_ok = []
        for s in specs:
            lockin = s.data[s.keys.LI]
            bias = s.data[s.keys.V]
            pts = np.array([
                lockin[bias.between(c.left, c.right)].mean()
                for c in self.conditions])
            pts -= np.min(pts)
            pts /= np.max(pts)

            pts_ok = (self._mins <= pts) & (pts <= self._maxs)
            self.pts_value.append(pts)
            self.pts_ok.append(pts_ok)
            self.identified.append(np.all(pts_ok))

    def show(self, channel, radius=0.8, **kwargs):
        ax = channel.plot(**kwargs)
        topo = channel.topo

        for spec, identified in zip(self.specs, self.identified):
            xy = topo.abs2topo(spec.xyz_nm[0:2])
            r = radius
            c = "Green" if identified else "Red"

            ax.add_patch(Circle(xy, r, fill=False, linewidth=3, color=c))

        return ax
