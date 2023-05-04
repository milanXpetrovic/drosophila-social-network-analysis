import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def plot_histogram(x, y, title, save_name):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=1000)
    heatmap = gaussian_filter(heatmap, sigma=16)

    plt.imshow(
        heatmap.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cm.jet,
    )
    plt.colorbar(label="Count")
    plt.title(title)

    plt.savefig(f"{save_name}.png")
    plt.savefig(f"{save_name}.svg", format="svg")
    plt.savefig(f"{save_name}.eps", format="eps")

    plt.close()
