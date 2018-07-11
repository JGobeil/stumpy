import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
from matplotlib.figure import Figure


heigh_figsize_name = {
    'Large': 20,
    'large': 15,
    'medium': 10,
    'small': 6,
    'tiny': 4,
    'default': 10,
}

shape_figsize_name = {
    'square': 1.0,
    'golden': 2 / (1 + 5**0.5),
    'default': 2 / (1 + 5**0.5),
}

_hfn = heigh_figsize_name
_sfn = shape_figsize_name

def get_figsize(size=None, shape=None) -> (float, float):
    if isinstance(size, tuple):
        return size
    else:
        if isinstance(size, float) or isinstance(size, int):
            h = size
        else:
            h = _hfn[size] if size in _hfn else _hfn['default']

        if isinstance(shape, float) or isinstance(shape, int):
            s = shape
        else:
            s = _sfn[shape] if shape in _sfn else _sfn['default']
        return (h, h*s)


def create_figure(size=None, shape=None, pyplot=True, dpi=100, **kwargs) -> Figure:
    figsize = get_figsize(size, shape)

    if pyplot:
        figure = plt.figure(figsize=figsize)
    else:
        figure = Figure(figsize=figsize, dpi=dpi)
        FigureCanvasAgg(figure)

    return figure


def no_ticks(ax):
    ax.axis('on')
    ax.tick_params(
        axis='both',  # changes apply to the x-axis and y-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off


def no_axis(ax):
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def no_grid(ax):
    ax.grid(False)


def add_title(ax, title):
    ax.set_title(title)
