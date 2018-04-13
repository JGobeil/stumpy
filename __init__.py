
sxm_plot_defaults = {
    'info': 'normal',
    'show_axis': False,
    'boxed': True,
    'size': 'medium',
    'pyplot': True,
    'dpi': 100,
    'cmap': 'Blues_r',
}

topo_info_formatter = {
    'normal': (
        "{sxm.name} - {channel.name}\n"
        "{topo.size_nm_str}/{topo.size_px_str}"
        " - "
        "{channel.current_str}@{channel.bias_str}\n"
        "{sxm.record_datetime}"
    ),
    "minimal": "",
    #        info == "minimal":
    #infostr = "%.3d - %s - %s" % (
    #    sxm.serie_number,  # filename
    #    sxm.size_nm_str,  # size
    #    sxm.bias_str,  # bias
    #)
}

# for easy import
# .sxm file
from .sxmfile import SxmFile
from .topodataset import SxmDataSet

# .dat file
from .datfile import BiasSpec

# Tranfer function class and function
from .s2pfile import TFDataSet
from .s2pfile import open_s2p

from .opener import Opener
