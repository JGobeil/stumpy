import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
from matplotlib.figure import Figure


def get_figsize(size) -> (float, float):
    if isinstance(size, tuple):
        figsize = size
    elif size == 'large':
        figsize = (10, 10)
    elif size == 'medium':
        figsize = (7, 7)
    elif size == 'small':
        figsize = (5, 5)
    elif size == 'tiny':
        figsize = (3, 3)
    else:
        print(size, 'is not recognized as a figure size.')
        figsize = (10, 10)
    return figsize


def create_figure(size=None, pyplot=True, dpi=100, **kwargs) -> Figure:
    if size is None:
        size = 'medium'
    figsize = get_figsize(size)

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
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        left='off',
        right='off',
        labelleft='off',
        labelbottom='off')  # labels along the bottom edge are off


def no_axis(ax):
    ax.axis('off')


def no_grid(ax):
    ax.grid(False)


def add_title(ax, title):
    ax.set_title(title)
