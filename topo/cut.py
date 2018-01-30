import numpy as np

from .topo import Topo
from .topo import topomod


@topomod(True)
def cut_by_px(src, x1, x2, y1, y2):
    """ Cut a topo using pixel values."""
    data = src.data[x1:x2, y1:y2]
    scale = np.array(data.shape) / src.size_px
    size_nm = scale * src.size_nm
    pos_nm = src.pos_nm + src.pxpos_to_realpos((x2 - x1, y2 - y1))

    return Topo(
        data=data,
        size_nm=size_nm,
        pos_nm=pos_nm,
    )


@topomod(True)
def cut_by_nm(src, x1, x2, y1, y2):
    """ Cut a topo using positions in nm."""
    shape = np.array(src.data.shape)
    ((px1, py1), (px2, py2)) = np.array(
        shape * ((x1, y1), (x2, y2)), dtype=int)

    data = src.data[px1:px2, py1:py2]

    return Topo(
        data=data,
        size_nm=src.size_nm * (data.shape / shape),
        pos_nm=src.pxpos_to_realpos((px2 - px1, py2 - py1)),
    )
