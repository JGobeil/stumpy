
defaults = {
    'topoplot': {
        'info': 'normal',
        'show_axis': False,
        'boxed': True,
        'size': 'medium',
        'pyplot': True,
        'dpi': 100,
        'cmap': 'Blues_r',
        'save': False,
        'tight': True,
        'savepath': '',
        'fontdict_title': None,
        'absolute_pos': False,
    },
    'channels': '',


}

topoinfo_formatter = {
    'normal': (
        "{sxm.name} - {channel.name} - #{channel.number}\n"
        "{topo.size_nm_str}/{topo.size_px_str}"
        " - "
        "{channel.current_str}@{channel.bias_str}\n"
        "{sxm.record_datetime}"
    ),
    "minimal": (
        "{sxm.serie_number:03} - {channel.name}\n"
        "{topo.size_nm_str} - "
        "{channel.current_str}@{channel.bias_str}"
        )
}

search_path = ['.', ]

# for easy import
# .sxm file
from .sxmfile import SxmFile
from .topo import Topo
from .topo import InterpolatedTopo
from .topo import TopoSet

# .dat file
from .datfile import BiasSpec
from .datfile import GenericDatFile