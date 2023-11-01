import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import numpy as np

GRADIENT_COLORS = ["#FF66C4", "#5CE1E6", "#8C52FF"]
GRADIENT_COLORMAP = pltcolors.LinearSegmentedColormap.from_list(
    "gradient_cmap", GRADIENT_COLORS
)


def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(
        1, n, figsize=(n * 2 + 2, 3), layout="constrained", squeeze=False
    )
    ax: plt.Axes
    cmap: pltcolors.Colormap
    for [ax, cmap] in zip(axs.flat, colormaps): # type: ignore
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()