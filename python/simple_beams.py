"""
File to contain some simple beam analysis formulas. Will aim to include a useful but not
exhaustive set of beam formulas.
"""

from inspect import ismethod

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
            the distance from end 1 at which the load starts. If None, taken to be at
            end 1 of the beam.
        :param load_value: Value of the point point load or start value of
            distributed load.
        :param load_end_location: Position from end 1 at which the load finishes. Only
                required for distributed loads. If None, taken to be at end 2 of the beam.
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

        if load_location is None:
            load_location = 0

        if load_location < 0:
            raise ValueError(
                f"Load is located at point {load_location} "
                + f"which is before the start of the beam at 0.0."
            )

        if load_location > self.length:
            raise ValueError(
                f"Load is located at point {load_location} "
                + f"which is beyond the end of the beam (length = {self.length})."
            )

        self.load_location = load_location

        self.load_value = load_value

        if self.load_type == "D":

            if load_end_location is None:
                load_end_location = self.length

            if self.load_end_location > self.length:
                raise ValueError(
                    f"Load end is located at point {load_end_location} "
                    + f"which is beyond the end of the beam (length = {self.length})."
                )

            if self.load_end_location < self.load_location:
                raise ValueError(
                    f"Load end is located at point {load_end_location} "
                    + f"which is before the start of the distributed load "
                    + f"(start location = {self.length})."
                )

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

        return self._property_flipper(reactions, flipped_parameter="R2")

    @property
    def R2(self):
        """
        The reaction at end 2.
        """

        reactions = {"UF": lambda: self.load_value}

        return self._property_flipper(reactions, flipped_parameter="R1")

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

        mmax = {"UF": lambda: self.load_value * (self.length - self.load_location)}

        return self._property_flipper(mmax, flipped_parameter="Mmax")

    def M(self, x):
        """
        The moment in the member calculated at point x.

        :param x: The distance along the member from point 1.
        """

        def uf_helper():

            lever = max(0, x - self.load_location)

            return self.load_value * lever

        m = {"UF": uf_helper}

        return self._property_flipper(m, "M", x)

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

    def _flipped_beam(self):
        """
        Generate a flipped / mirrored beam element with appropriate properties.
        To be used to allow values to be calculated where a beam has mirrored support
        conditions - e.g. to calculate values where the support condition is "FU", but
        the property tables etc. only have "UF" in them.
        """

        f1 = self.f2
        f2 = self.f1

        if self.load_type == "P":
            load_location = self.length - self.load_location
            load_value = self.load_value
            load_end_location = None
            load_end_value = None
        else:
            load_location = self.length - self.load_end_location
            load_value = self.load_end_value
            load_end_location = self.length - self.load_location
            load_end_value = self.load_value

        return SimpleBeam(
            f1=f1,
            f2=f2,
            length=self.length,
            load_type=self.load_type,
            load_location=load_location,
            load_value=load_value,
            load_end_location=load_end_location,
            load_end_value=load_end_value,
        )

    def _property_flipper(self, property_calcs, flipped_parameter, *args):
        """
        A helper method to allow properties to be defined based on one beam orientation
        only (e.g. "FU" support condition) but calculated for the inverse (e.g. "UF").

        :param property_dict: A dictionary containing helper functions to calculate the
            property. If being passed from a method with required arguments,
            all arguments should already be rolled into the helper functions as part
            of the definition of the functions (e.g. they should have no parameters).
        :param flipped_parameter: The parameter to query in the flipped condition (e.g.
            if you want R1, but for the flipped beam, you need to query R2.
        :param args: Any arguments to be used when calling a method on the flipped beam
            element. If a property is being called then don't provide any args.
        """

        if self.support_condition in property_calcs:
            return property_calcs[self.support_condition]()

        if self.support_condition[::-1] in property_calcs:

            flipped = self._flipped_beam()

            if ismethod(getattr(flipped, flipped_parameter)):
                return getattr(flipped, flipped_parameter)(*args)

            return getattr(flipped, flipped_parameter)

        raise NotImplementedError

    def __repr__(self):

        return (
            f"{type(self).__name__}, support conditions:"
            + f"{self.support_condition}"
            + ", length="
            + f"{self.length}"
            + ", load type="
            + f"{self.load_type}"
        )
