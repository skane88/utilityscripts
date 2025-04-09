"""
Create some custom plotting functions
"""

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
