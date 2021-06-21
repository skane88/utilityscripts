"""
File to contain some simple beam analysis formulas. Will aim to include a useful but not
exhaustive set of beam formulas.
"""


class SimpleBeam:
    """
    Defines a simple beam element. This is intended to be a parent class, with derived
    classes implementing the actual functionality.
    """

    def __init__(self, f1, f2, length):
        """
        Defines a simple beam element.

        1 <-----length-----> 2
        ______________________

        :param f1: Fixity at end 1. Should be 'F' for fixed, 'P' for pinned or 'U'
            for free / unrestrained
        :param f2: Fixity at end 2
        :param length: Length
        """

        self.f1 = f1
        self.f2 = f2
        self.length = length

    @property
    def R1(self):
        """
        The reaction at end 1.
        """

        raise NotImplementedError()

    @property
    def R2(self):
        """
        The reaction at end 2.
        """

        raise NotImplementedError()

    def Rmax(self, absolute: bool = True):
        """
        The maximum reaction.

        :param absolute: Return the absolute maximum? Will return the reaction at the end
            that has the highest reaction in an absolute sense. Note that the sign of the
            reaction will be preserved though - e.g. if R1=-100 and R2 = 60, the return
            value will be -100.
        """

        if absolute:
            if abs(self.R1) > abs(self.R2):
                return self.R1
            return self.R2

        return max(self.R1, self.R2)

    @property
    def Mmax(self):
        """
        The maximum moment in the member. Will be returned as a Tuple in the format
            (Mmax, location from 1)
        """

        raise NotImplementedError()

    @property
    def Vmax(self):
        """
        The maximum shear in the member. Will be returned as a Tuple in the format
            (Vmax, location from 1)
        """

        raise NotImplementedError()

    @property
    def Deltamax(self):
        """
        The maximum deflection in the member. Will be returned as a Tuple in the format
            (Deltamax, location from 1)
        """

        raise NotImplementedError()
