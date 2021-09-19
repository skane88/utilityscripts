"""
File to contain some simple beam analysis formulas. Will aim to include a useful but not
exhaustive set of beam formulas.
"""

from inspect import ismethod
from typing import Union

import numpy as np


class PointLoad:
    def __init__(self, magnitude, location=None):
        """
        Defines a point load.

        :param magnitude: The magnitude of the load.
        :param location: The location of the load, as a distance from end 1 of the beam.
            Units are a % of the span. If None, assumed to be 0.0.
        """

        self.magnitude = magnitude
        self.location = 0 if location is None else location


class DistributedLoad:
    def __init__(self, magnitude_a, location_a=None, magnitude_b=None, location_b=None):
        """
        Defines a distributed load.

        :param magnitude_a: The distributed load at the start of the load. In units of
            Force / Length.
        :param location_a: The location of the start of the load, as a distance from end 1
            of the beam. Units are a % of the span. If None, assumed to be 0.0.
        :param magnitude_b:  The distributed load at the end of the load. If None, assumed
            to be the same as magnitude_a. In units of Force / Length.
        :param location_b: The location of the end of the load, as a distance from end 1
            of the beam. Units are a % of the span. If None, assumed to be 1.0.
        """

        self.magnitude_a = magnitude_a

        if location_a is None:
            location_a = 0.0

        if location_a < 0:
            raise ValueError(f"location_a must be a positive value, got {location_a}.")

        if location_a > 1.0:
            raise ValueError(f"Expected location_a to be <1.0, got {location_a}")

        self.location_a = location_a

        if location_b is None:
            location_b = 1.0

        if location_b < location_a:
            raise ValueError(
                f"Expected location_b ({location_b}) to be greater "
                + f"than location_a ({location_a})"
            )

        if location_b > 1.0:
            raise ValueError(f"Expected location_a to be <1.0, got {location_a}")

        self.location_b = location_b

        self.magnitude_b = 0 if magnitude_b is None else magnitude_b


class SimpleBeam:
    """
    Defines a simple beam element.
    """

    def __init__(
        self, f1, f2, length, E, I, load: Union[PointLoad, DistributedLoad],
    ):
        """
        Defines a simple beam element.

        1 |<------length------>|2

        f1______________________f2

        |<----->x

        :param f1: Fixity at end 1. Should be 'F' for fixed, 'P' for pinned or 'U'
            for free / unrestrained
        :param f2: Fixity at end 2
        :param length: Length
        :param E: The elastic modulus of the beam.
        :param I: The second moment of inertia of the beam.
        :param load: A PointLoad or DistributedLoad object.
        """

        self.f1 = f1
        self.f2 = f2
        self.length = length
        self.E = E
        self.I = I
        self.load = load

    @property
    def load_type(self):

        return type(self.load).__name__

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

        def uf_helper():

            if isinstance(self.load, PointLoad):
                return self.load.magnitude

        reactions = {"UF": uf_helper}

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

        vmax = {"UF": lambda: self.load_value}

        return self._property_flipper(vmax, flipped_parameter="Vmax")

    def V(self, x):
        """
        The shear in the member calculated at point x.

        :param x: The distance along the member from point 1.
        """

        def uf_helper():
            if x < self.load_location:
                return 0

            return self.load_value

        m = {"UF": uf_helper}

        return self._property_flipper(m, "V", x)

    @property
    def Deltamax(self):
        """
        The maximum deflection in the member. Will be returned as a Tuple in the format
            (Deltamax, location from 1)
        """

        def ufhelper():

            P = self.load_value
            L = self.length
            b = L - self.load_location
            E = self.E
            I = self.I

            return ((P * b * b) / (6 * E * I)) * (3 * L - b)

        deltamax = {"UF": ufhelper}

        return self._property_flipper(deltamax, flipped_parameter="Deltamax")

    def Delta(self, x):
        """
        The deflection of the member calculated at point x.

        :param x: The distance along the member from point 1.
        """

        def ufhelper():
            P = self.load_value
            L = self.length
            a = self.load_location
            b = L - a
            E = self.E
            I = self.I

            if x < a:
                return ((P * b * b) / (6 * E * I)) * (3 * L - 3 * x - b)

            return ((P * (L - x) ** 2) / (6 * E * I)) * (3 * b - L + x)

        deltamax = {"UF": ufhelper}

        return self._property_flipper(deltamax, flipped_parameter="Delta")

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

        if isinstance(self.load, PointLoad):
            load = PointLoad(
                magnitude=self.load.magnitude, location=1.0 - self.load.location
            )

        else:
            load = DistributedLoad(
                magnitude_a=self.load.magnitude_b,
                magnitude_b=self.load.magnitude_a,
                location_a=1.0 - self.load.location_b,
                location_b=1.0 - self.load.location_a,
            )

        return SimpleBeam(
            f1=f1, f2=f2, E=self.E, I=self.I, length=self.length, load=load
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
