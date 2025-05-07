"""
Module to contain functions to check a lug.
"""

from math import radians

import numpy as np
import pandas as pd

from utilityscripts.lifting_lug.lifting_lug import Lug, LugLoad
from utilityscripts.steel.as4100 import s9_1_9_block, s9_2_2_4_v_bt, s9_2_2_4_v_by

ULS_STATIC = "uls_static"
ULS_DYNAMIC = "us_dynamic"
ALLOWABLE_STATIC = "allowable_static"
ALLOWABLE_DYNAMIC = "allowable_dynamic"
LOAD = "_load"
DESIGN_RATIO = "_design_ratio"


class Result:
    """
    A class to store results with some convenience methods.

    NOTE: This class makes no judgement on whether smaller or larger values are
    "better" or "worse". This is left up to the users of the class.
    """

    def __init__(
        self,
        *,
        in_plane_angles: list | np.ndarray,
        out_of_plane_angles: list | np.ndarray,
        uls_capacity: list | np.ndarray,
        allowable_capacity: list | np.ndarray,
        uls_static_load: list | np.ndarray,
        uls_dynamic_load: list | np.ndarray,
        allowable_static_load: list | np.ndarray,
        allowable_dynamic_load: list | np.ndarray,
        uls_static_design_ratio: list | np.ndarray,
        uls_dynamic_design_ratio: list | np.ndarray,
        allowable_static_design_ratio: list | np.ndarray,
        allowable_dynamic_design_ratio: list | np.ndarray,
    ):
        """
        Create a result object. This is simply a wrapper around a Pandas dataframe.
        """

        self._results = pd.DataFrame(
            data={
                "in_plane_angle": in_plane_angles,
                "out_of_plane_angle": out_of_plane_angles,
                "uls_capacity": uls_capacity,
                "allowable_capacity": allowable_capacity,
                ULS_STATIC + LOAD: uls_static_load,
                ULS_DYNAMIC + LOAD: uls_dynamic_load,
                ALLOWABLE_STATIC + LOAD: allowable_static_load,
                ALLOWABLE_DYNAMIC + LOAD: allowable_dynamic_load,
                ULS_STATIC + DESIGN_RATIO: uls_static_design_ratio,
                ULS_DYNAMIC + DESIGN_RATIO: uls_dynamic_design_ratio,
                ALLOWABLE_STATIC + DESIGN_RATIO: allowable_static_design_ratio,
                ALLOWABLE_DYNAMIC + DESIGN_RATIO: allowable_dynamic_design_ratio,
            },
        )

    @property
    def in_plane_angles(self) -> np.ndarray:
        """
        The angles in-plane for which the results have been calculated.

        Returns
        -------
        np.ndarray
            A numpy array containing sorted, unique values.
            There is no guarantee that the original input data was sorted or unique.
        """

        return np.sort(pd.unique(self.results.in_plane_angle))

    @property
    def out_of_plane_angles(self) -> np.ndarray:
        """
        The angles out-of-plane for which the results have been calculated.

        Returns
        -------
        np.ndarray
            A numpy array containing sorted, unique values.
            There is no guarantee that the original input data was sorted or unique.
        """

        return np.sort(pd.unique(self.results.out_of_plane_angle))

    @property
    def results(self) -> pd.DataFrame:
        """
        The results.

        Returns
        -------
        pd.DataFrame
            A Pandas dataframe containing the results, with columns for
            the angles at which they were calculated.
        """

        return self._results

    @property
    def no_results(self) -> int:
        """
        The number of results.
        """

        return self.results.in_plane_angle.size

    @property
    def in_plane_increments(self) -> int:
        """
        The number of in-plane angles considered.
        """

        return self.in_plane_angles.size

    @property
    def out_of_plane_increments(self) -> int:
        """
        The number of out-of-plane angles considered
        """

        return self.out_of_plane_angles.size

    @property
    def min_in_plane_angle(self):
        """
        The minimum in-plane angle which was assessed.
        """

        return self.results.in_plane_angle.min()

    @property
    def max_in_plane_angle(self):
        """
        The maximum in-plane angle which was assessed.
        """

        return self.results.in_plane_angle.max()

    @property
    def min_out_of_plane_angle(self):
        """
        The minimum out-of-plane angle which was assessed.
        """

        return self.results.out_of_plane_angle.min()

    @property
    def max_out_of_plane_angle(self):
        """
        The maximum out-of-plane angle which was assessed.
        """
        return self.results.out_of_plane_angle.max()

    @property
    def min_uls_capacity(self):
        """
        The minimum ULS capacity
        """
        return self.results.uls_capacity.min()

    @property
    def max_uls_capacity(self):
        """
        The maximum ULS capacity.
        """
        return self.results.uls_capacity.max()

    @property
    def min_allowable_capacity(self):
        """
        The minimum Allowable capacity.
        """
        return self.results.allowable_capacity.min()

    @property
    def max_allowable_capacity(self):
        """
        The Maximum allowable capacity.
        """
        return self.results.allowable_capacity.max()

    def _result_chooser(
        self, *, uls: bool | None, dynamic: bool | None, load: bool
    ) -> list[str]:
        """
        Choose the results column to choose from. If either uls or dynamic is None,
        returns all applicable columns.

        Parameters
        ----------
        uls : bool | None
            return uls loads if True, else allowable.
        dynamic : bool | None
            return dynamic results if True, else static.
        load : bool
            Return load columns if true, otherwise return design ratios
        """

        col_dict = {
            (None, None): [
                ULS_STATIC,
                ULS_DYNAMIC,
                ALLOWABLE_STATIC,
                ALLOWABLE_DYNAMIC,
            ],
            (None, True): [ULS_DYNAMIC, ALLOWABLE_DYNAMIC],
            (None, False): [ULS_STATIC, ALLOWABLE_STATIC],
            (True, None): [ULS_STATIC, ULS_DYNAMIC],
            (False, None): [ALLOWABLE_STATIC, ALLOWABLE_DYNAMIC],
            (True, True): [ULS_DYNAMIC],
            (True, False): [ULS_STATIC],
            (False, True): [ALLOWABLE_DYNAMIC],
            (False, False): [ALLOWABLE_STATIC],
        }

        cols = col_dict[(uls, dynamic)]

        return [c + LOAD for c in cols] if load else [c + DESIGN_RATIO for c in cols]

    def min_load(self, *, uls: bool | None = True, dynamic: bool | None = True):
        """
        Get the minimum load applied to the load cases.

        Parameters
        ----------
        uls : bool | None
            Return ULS results? If False returns allowable,
            if None returns both ULS and Allowable.
        dynamic : bool | None
            Return dynamic results? If False returns static,
            if None returns both static & dynamic.
        """

        columns = self._result_chooser(uls=uls, dynamic=dynamic, load=True)
        return self.results[columns].min(axis=None)

    def max_load(self, *, uls: bool | None = True, dynamic: bool | None = True):
        """
        Get the maximum load applied to the load cases.

        Parameters
        ----------
        uls : bool | None
            Return ULS results? If False returns allowable,
            if None returns both ULS and Allowable.
        dynamic : bool | None
            Return dynamic results? If False returns static,
            if None returns both static & dynamic.
        """

        columns = self._result_chooser(uls=uls, dynamic=dynamic, load=True)
        return self.results[columns].max(axis=None)

    def min_design_ratio(self, *, uls: bool | None = True, dynamic: bool | None = True):
        """
        Get the minimum design ratio in the load cases considered.

        Parameters
        ----------
        uls : bool | None
            Return ULS results? If False returns allowable,
            if None returns both ULS and Allowable.
        dynamic : bool | None
            Return dynamic results? If False returns static,
            if None returns both static & dynamic.
        """

        columns = self._result_chooser(uls=uls, dynamic=dynamic, load=False)
        return self.results[columns].min(axis=None)

    def max_design_ratio(self, *, uls: bool | None = True, dynamic: bool | None = True):
        """
        Get the maximum design ratio in the cases considered.

        Parameters
        ----------
        uls : bool | None
            Return ULS results? If False returns allowable,
            if None returns both ULS and Allowable.
        dynamic : bool | None
            Return dynamic results? If False returns static,
            if None returns both static & dynamic.
        """

        columns = self._result_chooser(uls=uls, dynamic=dynamic, load=False)

        return self.results[columns].max(axis=None)

    def __repr__(self):
        return (
            f"{type(self).__name__}: "
            + f"with {self.no_results} total results."
            + " Min / max design ratios: "
            + f"{self.min_design_ratio(uls=None, dynamic=None):.2f}"
            + f"/{self.max_design_ratio(uls=None, dynamic=None):.2f}"
            + f" Max design load {self.max_load(uls=None, dynamic=None):.2e}."
        )


class LugCheck:
    """
    Class to carry out the actual check. The intent behind having a separate class
    is so that if it makes sense in future additional LugCheck classes could be
    created for different design codes etc.
    """

    def __init__(
        self,
        *,
        lug: Lug,
        loads: LugLoad | list[LugLoad] | dict[str, LugLoad],
        in_plane_increments=101,
        out_of_plane_increments: int = 101,
        phi_steel=0.90,
    ):
        """
        Parameters
        ----------
        lug : Lug
            The Lug to check.
        loads : LugLoad | list[LugLoad] | dict[str, LugLoad]
            The loads to check the lug for. Given as a dictionary:
            {load case name: LugLoad}
            Will also accept a single LugLoad or a list of LugLoads, and will
            generate load case names of the format: "Load Case i"
        in_plane_increments : int
            The no. of in-plane increments to assess each
            load case against.
        out_of_plane_increments : int
            The no. of out-of-plane increments to assess each load case against.
        phi_steel : float
            The capacity reduction factor for steel to use in checking the lug.
        """

        self._lug = lug
        self._in_plane_increments = in_plane_increments
        self._out_of_plane_increments = out_of_plane_increments
        self._phi_steel = phi_steel

        if isinstance(loads, LugLoad):
            self._loads = {"Load Case 1": loads}
            self._load_index = {0: "Load Case 1"}
        elif isinstance(loads, list):
            self._loads = {}
            self._load_index = {}

            for i, load in enumerate(loads):
                key = f"Load Case {i + 1}"

                self._loads[key] = load
                self._load_index[i] = key

        else:
            self._loads = loads
            self._load_index = {}

            for i, key in enumerate(loads):
                self._load_index[i] = key

        self._results = {key: {} for key in self.loads}

    @property
    def lug(self) -> Lug:
        """
        The lug being checked.
        """

        return self._lug

    @property
    def in_plane_increments(self) -> int:
        """
        The default no. of out-of-plane angles to assess the lug against.
        """

        return self._in_plane_increments

    @property
    def out_of_plane_increments(self) -> int:
        """
        The default no. of out-of-plane angles to assess the lug against.
        """

        return self._out_of_plane_increments

    @property
    def phi_steel(self):
        """
        The capacity reduction factor for steel used to check the lug.
        """

        return self._phi_steel

    @property
    def loads(self) -> dict[str, LugLoad]:
        """
        The loads to check the lug for. Given as a dictionary of the format:
        {load case name: LugLoad}
        """

        return self._loads

    @property
    def load_index(self) -> dict[int, str]:
        """
        Index that matches the load case title used for the load, with the
        load case string.
        """

        return self._load_index

    @property
    def load_cases(self) -> list[str]:
        """
        A list of the load case names.
        """

        return list(self.loads)

    @property
    def no_loads(self) -> float:
        """
        The no. of load cases being checked.
        """

        return len(self.loads)

    def _get_load(self, load: int | str) -> LugLoad:
        """
        Helper method to get a load regardless of if the load case key or the
        integer key is passed.

        Parameters
        ----------
        load : int | str
            The load to get, either as a string with the case name,
            or the integer index to the load.
        """

        if isinstance(load, str):
            return self.loads[load]

        return self.loads[self._load_index[load]]

    def bearing_yield_capacity_single(
        self,
        *,
        load: LugLoad,
        phi: float,
        **kwargs,
    ):
        """
        Calculate the bearing yield capacity of the lug for a given load case.
        This function returns a single number.

        Parameters
        ----------
        load : LugLoad
            The load to assess the lug against.
        phi : float
            The capacity reduction factor for ULS checks. Use 1.0 for
            Allowable Stress checks.
        kwargs : dict
            Provided so this function can be used as a capacity function
            when creating a Result object. Not necessary otherwise.
        """

        return phi * s9_2_2_4_v_by(
            d_f=load.dia_pin,
            t_p=self.lug.thickness,
            f_up=self.lug.f_up,
        )

    def bearing_tearout_capacity_single(
        self,
        *,
        in_plane_angle: float,
        load: LugLoad,
        phi: float,
        **kwargs,
    ):
        """
        Calculates the bearing tearout capacity at a given angle.

        When calculating the edge distance, the minimum edge distance within +/-15deg
        is considered. The choice of 10deg is based on engineering judgement only.

        Parameters
        ----------
        load : LugLoad
            The load to assess the lug against.
        in_plane_angle : float
            The angle to assess the tearout capacity for.
        phi : float
            The capacity reduction factor for the steel for ULS checks.
            Typically, use 1.0 for Allowable capacities.
        kwargs : dict
            Only included so this function can be used as a
            capacity function when creating a Result object. Not necessary otherwise.
        """

        angle_tol = radians(10)
        dia_pin = load.dia_pin

        # note: because the edge distance increases / decreases smoothly,
        # only need to check the min, max and central angles.
        check_angles = [
            in_plane_angle - angle_tol,
            in_plane_angle,
            in_plane_angle + angle_tol,
        ]

        a_e = (
            np.min(np.asarray([self.lug.slice(angle=a).length for a in check_angles]))
            + dia_pin / 2
        )

        return phi * s9_2_2_4_v_bt(a_e=a_e, t_p=self.lug.thickness, f_up=self.lug.f_up)

    def block_shear_capacity_single(
        self,
        *,
        in_plane_angle: float,
        phi: float,
        **kwargs,
    ):
        """
        Calculates the bearing tearout capacity at a given angle.

        When calculating the edge distance, the minimum edge distance within +/-15deg
        is considered. The choice of 10deg is based on engineering judgement only.

        Parameters
        ----------
        load : LugLoad
            The load to assess the lug against.
        in_plane_angle : float
            The angle to assess the tearout capacity for.
        phi : float
            The capacity reduction factor for the steel for ULS checks.
            Note that this value is usually 0.75 according to AS4100.
            Typically, use 1.0 for Allowable capacities.
        kwargs : dict
            Only included so this function can be used as a
            capacity function when creating a Result object. Not necessary otherwise.
        """

        pos_tension_angle = in_plane_angle + radians(90)
        neg_tension_angle = in_plane_angle - radians(90)

        pos_tension_net = self.lug.cut(angle=pos_tension_angle, holes=True)
        neg_tension_net = self.lug.cut(angle=neg_tension_angle, holes=True)

        tension_net = min(pos_tension_net, neg_tension_net)

        shear_gross = self.lug.cut(angle=in_plane_angle, holes=False)
        shear_net = self.lug.cut(angle=in_plane_angle, holes=True)

        a_gv = shear_gross * self.lug.thickness
        a_nv = shear_net * self.lug.thickness
        a_nt = tension_net * self.lug.thickness

        k_bs = 0.5

        return phi * s9_1_9_block(
            a_gv=a_gv,
            a_nv=a_nv,
            a_nt=a_nt,
            f_yp=self.lug.f_yp,
            f_up=self.lug.f_up,
            k_bs=k_bs,
        )

    def _result_creator(self, capacity_func, phi, load_case: str | int):
        """
        Function to build a results object.

        Done separately from the individual design checks,
        so that common code can be re-used.

        Parameters
        ----------
        capacity_func : callable
            A capacity function, with 4x arguments or
            **kwargs where the arguments are ignored:
            load, in_plane_angle, out_of_plane_angle and phi
        phi : float
            The capacity reduction factor.
        load_case : str | int
            The load case to get the results for.
        """

        load = self._get_load(load=load_case)

        in_plane = load.in_plane_angles(no_increments=self.in_plane_increments)
        out_plane = load.out_of_plane_angles(no_increments=self.out_of_plane_increments)

        # note: can't simply use self.no_increments because if the min & max angles are
        # the same then the load is smart enough to only return a single one.
        in_plane_increments = in_plane.size
        out_plane_increments = out_plane.size

        # now tile the
        in_plane = np.tile(
            in_plane,
            out_plane_increments,
        )
        out_plane = np.concatenate(
            [np.full(shape=in_plane_increments, fill_value=a) for a in out_plane]
        )

        total_increments = in_plane_increments * out_plane_increments

        uls_capacity = np.zeros(shape=total_increments)
        allowable_capacity = np.zeros(shape=total_increments)
        uls_load_static = np.zeros(shape=total_increments)
        uls_load_dynamic = np.zeros(shape=total_increments)
        allowable_load_static = np.zeros(shape=total_increments)
        allowable_load_dynamic = np.zeros(shape=total_increments)

        for i, angles in enumerate(zip(in_plane, out_plane, strict=True)):
            in_plane_angle = angles[0]
            out_plane_angle = angles[1]

            load_component = load.generate_single_load(
                in_plane_angle=in_plane_angle, out_of_plane_angle=out_plane_angle
            )

            uls_capacity[i] = capacity_func(
                load=load,
                in_plane_angle=in_plane_angle,
                out_of_plane_angle=out_plane_angle,
                phi=phi,
            )
            allowable_capacity[i] = capacity_func(
                load=load,
                in_plane_angle=in_plane_angle,
                out_of_plan_angle=out_plane_angle,
                phi=1.0,
            )

            uls_load_static[i] = load_component.load_uls_static
            uls_load_dynamic[i] = load_component.load_uls_dynamic
            allowable_load_static[i] = load_component.load
            allowable_load_dynamic[i] = load_component.load_dynamic

        design_ratio_uls_static = uls_load_static / uls_capacity
        design_ratio_uls_dynamic = uls_load_dynamic / uls_capacity
        design_ratio_allowable_static = allowable_load_static / allowable_capacity
        design_ratio_allowable_dynamic = allowable_load_dynamic / allowable_capacity

        return Result(
            in_plane_angles=in_plane,
            out_of_plane_angles=out_plane,
            uls_capacity=uls_capacity,
            allowable_capacity=allowable_capacity,
            uls_static_load=uls_load_static,
            uls_dynamic_load=uls_load_dynamic,
            allowable_static_load=allowable_load_static,
            allowable_dynamic_load=allowable_load_dynamic,
            uls_static_design_ratio=design_ratio_uls_static,
            uls_dynamic_design_ratio=design_ratio_uls_dynamic,
            allowable_static_design_ratio=design_ratio_allowable_static,
            allowable_dynamic_design_ratio=design_ratio_allowable_dynamic,
        )

    def bearing_yield_capacity(
        self,
        load_case: str | int,
    ) -> Result:
        """
        Calculate the bearing yield capacity of the lug against all valid angles
        in a given load case.

        Parameters
        ----------
        load_case : str | int
            The load case to assess.
        """

        return self._result_creator(
            capacity_func=self.bearing_yield_capacity_single,
            load_case=load_case,
            phi=self.phi_steel,
        )

    def bearing_tearout_capacity(self, load_case: str | int):
        """
        Calculate the bearing tearout capacity against all valid angles in a load case.

        Parameters
        ----------
        load_case : str | int
            The load case to assess.
        """

        return self._result_creator(
            capacity_func=self.bearing_tearout_capacity_single,
            load_case=load_case,
            phi=self.phi_steel,
        )

    def block_shear_capacity(self, load_case: str | int):
        """
        Calculate the block shear capacity against all valid angles in a load case.

        Parameters
        ----------
        load_case : str | int
            The load case to assess.
        """

        return self._result_creator(
            capacity_func=self.block_shear_capacity_single,
            phi=0.75,
            load_case=load_case,
        )

    def __repr__(self):
        return f"{type(self).__name__}" + f" with {self.no_loads} load cases."
