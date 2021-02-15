"""
File to contain some basic AS1170.2 helper methods for working with wind loads
"""

from pathlib import Path

import toml

FILE_PATH = Path(__file__)
DEFAULT_DATA_PATH = FILE_PATH.parent / Path("as1170_2.toml")
STANDARD_DATA = {}


def init_standard_data(*, file_path=None):
    """
    Initialise the standard_data dictionary if required.
    :param file_path: An optional filepath rather than using DEFAULT_DATA
    """

    global STANDARD_DATA

    if file_path is None:
        file_path = DEFAULT_DATA_PATH

    STANDARD_DATA = toml.load(f=file_path)


def V_R_basic(*, a, b, R, k):
    """
    Calculate the basic windspeed for a wind region. Ignores parameters F_C or F_D, for those use method V_R

    :param a: Windspeed parameter 'a'
    :param b: Windspeed parameter 'b'
    :param R: The Average Recurrence Interval (ARI) of the windspeed.
    :param k: Windspeed parameter 'k'
    """

    return a - b * R ** -k


def F_x(*, wind_region, R):
    """
    Calculate the cyclonic region factor F_C or F_D.

    :param wind_region: The wind region.
    :param R: The Average Recurrence Interval (ARI) of the windspeed.
    """

    F_x_val = STANDARD_DATA["region_windspeed_parameters"][wind_region]["F_x"]
    F_x_min_R = STANDARD_DATA["region_windspeed_parameters"][wind_region]["F_x_min_R"]

    if R < F_x_min_R:
        return 1.0

    return F_x_val


def V_R(*, wind_region: str, R, ignore_F_x: bool = False):
    """
    Calculate the regional windspeed V_R based on the region and return period.

    :param wind_region: The wind region where the structure is located.
    :param R: The Average Recurrence Interval (ARI) of the windspeed.
    :param ignore_F_x: Ignore the cyclonic region factor F_C or F_D?
    :return: The regional windspeed.
    """

    if len(STANDARD_DATA) == 0:
        init_standard_data()

    if not ignore_F_x:
        F = F_x(wind_region=wind_region, R=R)
    else:
        F = 1.0

    a = STANDARD_DATA["region_windspeed_parameters"][wind_region]["a"]
    b = STANDARD_DATA["region_windspeed_parameters"][wind_region]["b"]
    k = STANDARD_DATA["region_windspeed_parameters"][wind_region]["k"]
    V_min = STANDARD_DATA["region_windspeed_parameters"][wind_region]["V_min"]

    return max(V_min, F * V_R_basic(a=a, b=b, R=R, k=k))
