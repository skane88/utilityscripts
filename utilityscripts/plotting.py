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
    color=[
        "#333333",
        "#CE190D",
        "#4183C4",
        "#707070",
        "#AA3E98",
        "#F2A71B",
        "#268F63",
        "#E8E8E8",
    ]
)
PERSONAL_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL", ["#CE190D", "#4183C4"]
)
PERSONAL_RED_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL_RED", ["#CE190D", "#FFFFFF"]
)
PERSONAL_BLUE_COLORMAP = LinearSegmentedColormap.from_list(
    "PERSONAL_BLUE", ["#4183C4", "#FFFFFF"]
)
