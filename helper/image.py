import numpy as np
import matplotlib.cm as mplcm

from .log import get_logger
log = get_logger(__name__)


def writetopo_cv2(data, filename, cmap):
    """ Save a topo image using opencv and matplotlib colormap."""
    cm = mplcm.get_cmap(cmap)

    data = data.copy()
    data -= data.min()
    data *= (1/(data.max() - data.min()))
    cv2.imwrite(
        filename,
        cv2.cvtColor(cm(data).astype(np.float32)*255, cv2.COLOR_RGBA2BGR)
    )

def encodetopo_cv2(data, cmap):
    """ Save a topo image using opencv and matplotlib colormap."""
    cm = mplcm.get_cmap(cmap)

    data = data.copy()
    data -= data.min()
    data *= (1/(data.max() - data.min()))
    retval, buf =  cv2.imencode(
        '.png',
        cv2.cvtColor(cm(data).astype(np.float32)*255, cv2.COLOR_RGBA2BGR)
    )
    return buf


def writetopo_imageio(data, filename, cmap):
    """ Save a topo image using imageio and matplotlib colormap."""
    cm = mplcm.get_cmap(cmap)

    data = data.copy()
    data -= data.min()
    data *= (1/(data.max() - data.min()))
    imageio.imwrite(
        filename,
        (cm(data)*255).astype(np.uint8),
    )

def writetopo(data, filename, cmap):
    """ Save a topo image using opencv or imageio and
     matplotlib colormap."""
    log.err("Not able to load image module. Install opencv or imageio.")

try:
    import imageio
    writetopo = writetopo_imageio
    encodetopo = lambda *args: "undefined"
    print('imageio imported')
except ImportError:
    pass

try:
    import cv2
    writetopo = writetopo_cv2
    encodetopo = encodetopo_cv2
    print('cv2 imported')
except ImportError:
    pass

