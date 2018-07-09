import numpy as np

from .topo import Topo


class TopoCutted(Topo):
    def __init__(self, src, x1, x2, y1, y2, unit='nm'):
        super().__init__(src)
        self.limits = [x1, x2, y1, y2]
        self.size_nm = np.array((x2 - x1, y2 - y1))

        print(x1, x2, y1, y2)

        px1, px2 = np.array(np.array((x1, x2))*src.size_px[0]/src.size_nm[0],
                            dtype=int)

        py1, py2 = np.array(np.array((y1, y2))*src.size_px[1]/src.size_nm[1],
                            dtype=int)

        self.limits_nm = (x1, x2, y1, y2)
        self.limits_px = (px1, px2, py1, py2)

        self.pos_nm = src.pos_nm + (src.size_nm - np.array((x2 + x1, y2 + y1)))/2


    @property
    def data(self):
        px1, px2, py1, py2 = self.limits_px
        return self.src.data[px1:px2, py1:py2]


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
