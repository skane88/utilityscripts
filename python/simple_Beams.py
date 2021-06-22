"""
File to contain some simple beam analysis formulas. Will aim to include a useful but not
exhaustive set of beam formulas.
"""

import numpy as np


class SimpleBeam:
    """
    Defines a simple beam element.
    """

    def __init__(
        self,
        f1,
        f2,
        length,
        load_type,
        load_location,
        load_value,
        load_end_location=None,
        load_end_value=None,
    ):
        """
        Defines a simple beam element.

        1 <-----length-----> 2
        ______________________

        :param f1: Fixity at end 1. Should be 'F' for fixed, 'P' for pinned or 'U'
            for free / unrestrained
        :param f2: Fixity at end 2
        :param length: Length
        :param load_type: 'P' for point or 'D' for distributed.
        :param load_location: Point load location or start location of distributed load as
            the distance from end 1 at which the load starts
        :param load_value: Value of the point point load or start value of
            distributed load.
        :param load_end_location: Position from end 1 at which the load finishes. Only
                required for distributed loads.
        :param load_end_value: End value of distributed load. Only used for distributed
            loads, and if None the distributed load is taken to be constant.
        """

        self.f1 = f1
        self.f2 = f2
        self.length = length

        if load_type not in {"P", "D"}:
            raise ValueError(
                f"Invalid load type {load_type}. "
                + "Expected either a point 'P' or distributed load 'D'"
            )

        self.load_type = load_type
        self.load_location = load_location
        self.load_value = load_value

        if self.load_type == "D":
            self.load_end_location = load_end_location

            if load_end_value is None:
                self.load_end_value = self.load_value
            else:
                self.load_end_value = load_end_value

        else:
            self.load_end_location = None
            self.load_end_value = None

    @property
    def support_condition(self):
        """
        Returns the support conditions of the beam.
        """

        return self.f1 + self.f2

    @property
    def R1(self):
        """
        The reaction at end 1.
        """

        reactions = {"UF": lambda: 0.0}

        if self.support_condition in reactions:
            return reactions[self.support_condition]()

        raise NotImplementedError

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

    def M(self, x):
        """
        The moment in the member calculated at point x.

        :param x: The distance along the member from point 1.
        """

        raise NotImplementedError()

    @property
    def Vmax(self):
        """
        The maximum shear in the member. Will be returned as a Tuple in the format
            (Vmax, location from 1)
        """

        raise NotImplementedError()

    def V(self, x):
        """
        The shear in the member calculated at point x.

        :param x: The distance along the member from point 1.
        """

        raise NotImplementedError()

    @property
    def Deltamax(self):
        """
        The maximum deflection in the member. Will be returned as a Tuple in the format
            (Deltamax, location from 1)
        """

        raise NotImplementedError()

    def Delta(self, x):
        """
        The deflection of the member calculated at point x.

        :param x: The distance along the member from point 1.
        """

        raise NotImplementedError()

    def _invert_x(self, x):
        """
        A helper function to invert the position along the member (x) in cases where the
        support condition is reversed from the expected.
        """

        return self.length - x
