"""
Create some custom plotting functions
"""

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

AGILITUS_COLORS = cycler(
    color=["#0063ff", "#80f076", "#221f20", "#40a9bb", "#bec2cb", "#f79646"]
)
AGILITUS_COLORMAP = LinearSegmentedColormap.from_list(
    "AGILITUS", ["#0063ff", "#40a9bb", "#80f076"]
)
AGILITUS_BLUE_COLORMAP = LinearSegmentedColormap.from_list(
    "AGILITUS_BLUE", ["#0063ff", "#ffffff"]
)

PERSONAL_COLORS = cycler(
    color=[
        "#333333",
        "#CE190D",
        "#0D79CE",
        "#FDB813",
        "#79C99E",
        "#C69DD2",
        "#A8C256",
        "#E8E8E8",
    ]
)
PERSONAL_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL", ["#CE190D", "#0D79CE"]
)
PERSONAL_RED_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL_RED", ["#CE190D", "#FFFFFF"]
)
PERSONAL_BLUE_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL_BLUE", ["#0D79CE", "#FFFFFF"]
)


def plot_colormap_examples():
    """
    Create example plots for all defined colormaps.
    """

    # Create sample data
    x_base = np.linspace(0, 10, 100)
    y_base = np.linspace(0, 10, 100)
    x, y = np.meshgrid(x_base, y_base)
    z = np.sin(x) * np.cos(y)

    # Create figure with subplots for each colormap
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    fig.suptitle("Colormap Examples")

    # Plot each colormap
    axs[0, 0].pcolormesh(x, y, z, cmap=AGILITUS_COLORMAP)
    axs[0, 0].set_title("AGILITUS_COLORMAP")

    axs[0, 1].pcolormesh(x, y, z, cmap=AGILITUS_BLUE_COLORMAP)
    axs[0, 1].set_title("AGILITUS_BLUE_COLORMAP")

    axs[1, 0].pcolormesh(x, y, z, cmap=PERSONAL_COLORMAP)
    axs[1, 0].set_title("PERSONAL_COLORMAP")

    axs[1, 1].pcolormesh(x, y, z, cmap=PERSONAL_RED_COLORMAP)
    axs[1, 1].set_title("PERSONAL_RED_COLORMAP")

    axs[2, 0].pcolormesh(x, y, z, cmap=PERSONAL_BLUE_COLORMAP)
    axs[2, 0].set_title("PERSONAL_BLUE_COLORMAP")

    # Remove empty subplot
    fig.delaxes(axs[2, 1])

    plt.tight_layout()
    plt.show()


def plot_color_cycle_examples():
    """Create example plots for defined color cycles."""
    import numpy as np

    # Create sample data
    x = np.linspace(0, 10, 200)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(2 * x)
    y4 = np.cos(2 * x)
    y5 = np.sin(3 * x)
    y6 = np.cos(3 * x)
    y7 = np.sin(4 * x)
    y8 = np.cos(4 * x)

    # Create figure with subplots for each color cycle
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle("Color Cycle Examples")

    # Plot using AGILITUS colors
    ax1.set_prop_cycle(AGILITUS_COLORS)
    ax1.plot(x, y1, label="sin(x)")
    ax1.plot(x, y2, label="cos(x)")
    ax1.plot(x, y3, label="sin(2x)")
    ax1.plot(x, y4, label="cos(2x)")
    ax1.plot(x, y5, label="sin(3x)")
    ax1.plot(x, y6, label="cos(3x)")
    ax1.set_title("AGILITUS Colors")
    ax1.legend()

    # Plot using PERSONAL colors
    ax2.set_prop_cycle(PERSONAL_COLORS)
    ax2.plot(x, y1, label="sin(x)")
    ax2.plot(x, y2, label="cos(x)")
    ax2.plot(x, y3, label="sin(2x)")
    ax2.plot(x, y4, label="cos(2x)")
    ax2.plot(x, y5, label="sin(3x)")
    ax2.plot(x, y6, label="cos(3x)")
    ax2.plot(x, y7, label="sin(4x)")
    ax2.plot(x, y8, label="cos(4x)")
    ax2.set_title("PERSONAL Colors")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_colormap_examples()
    plot_color_cycle_examples()
