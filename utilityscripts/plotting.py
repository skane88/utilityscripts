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

PERSONAL_COLORS = cycler(
    color=["#333333", "#CE190D", "#1B365D", "#707070", "#A3D5FF", "#F2A71B", "#268F63"]
)
PERSONAL_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL", ["#CE190D", "#707070", "#1B365D"]
)
PERSONAL_RED_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL_RED", ["#CE190D", "#FFFFFF"]
)
PERSONAL_BLUE_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL_BLUE", ["#1B365D", "#FFFFFF"]
)
